import os
import json
import asyncio
import random
import re
import traceback
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区 =================
MODEL_NAME = "Qwen3-4B-ascend-flagos"
# MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
CONCURRENCY_LIMIT = 1
FEW_SHOT_K = 5  # 代码较长，Few-shot 数量不宜过多，防止超上下文

file_name_without_ext = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}.jsonl"
OUTPUT_FILE_V1 = f"./result/{MODEL_NAME}/{file_name_without_ext}-v1.jsonl"
ERROR_OUTPUT_FILE = f"./result/{MODEL_NAME}/{file_name_without_ext}_errors.jsonl"

# DATA_FILE = "../../../data/openseek-8_kernel_generation.json"
DATA_FILE = "../../../flag_scale/flag-os-3/LongContext-ICL-Annotation/data/openseek-8_kernel_generation.json"

DEFAULT_VALUE = ""

# 重试策略：第一次最贪婪(0.0)，后续逐步增加多样性以跳出错误格式
TEMPERATURE_STEPS = [0.0, 0.7, 1.0]
MAX_RETRIES = len(TEMPERATURE_STEPS) - 1
# ==========================================

# ================= Prompt =================
system_prompt = '''
You are an expert in writing efficient Triton GPU operators.
You will be given a detailed natural language description of a Triton kernel and its corresponding Python wrapper.

Your task:
- Generate correct, complete, and executable Triton + PyTorch code.
- Include both kernel(s) and Python wrapper function(s).
- Ensure the implementation strictly follows the described behavior.

---------------------
STRICT REQUIREMENTS
---------------------
1. Output ONLY the final code.
2. Do NOT include explanations, comments outside code, or extra text.
3. Do NOT include markdown formatting (NO ```python or ```).
4. Code must be clean, correct, and runnable.
5. 你的思考过程（<think>思考过程</think>）请限制在2000个字以内。

---------------------
STYLE GUIDELINES
---------------------
- Use `@triton.jit` for kernels.
- Use meaningful variable names consistent with the description.
- Ensure correct memory access (masking, strides, offsets).
- Handle edge cases (e.g., out-of-bounds with masks).
- Use efficient parallelization via grid definitions.
- Keep tensor layouts and contiguity in mind.
'''
# ==========================================

client = AsyncOpenAI(
    # api_key="EMPTY",
    # base_url="http://127.0.0.1:9010/v1",
    api_key="ms-c429b084-79ba-4a00-a749-aae8681e902d",
    base_url="https://api-inference.modelscope.cn/v1",
)


# ================= 技巧 1: 动态 Few-Shot 检索 =================

def tokenize(text):
    """简单的分词器，用于文本描述的 N-gram/关键词 Overlap 匹配"""
    return set(re.findall(r'\b\w+\b', str(text).lower()))


def retrieve_dynamic_few_shots(target_sample, pool, k=FEW_SHOT_K):
    """
    SimICL 检索算法：从 examples 中检索最相似的任务描述
    """
    if not pool or k == 0:
        return []

    target_tokens = tokenize(target_sample.get("input", ""))
    scored_pool = []

    for ex in pool:
        if ex.get("id") == target_sample.get("id") or not ex.get("output"):
            continue
        ex_tokens = tokenize(ex.get("input", ""))
        score = len(target_tokens & ex_tokens)
        scored_pool.append((score, ex))

    scored_pool.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_pool[:k]]


# ================= 技巧 3: Prompt & 格式对齐 =================

def format_few_shot_assistant(example):
    """提取 Few-shot 中的代码作为规范示例"""
    gt = example.get("output", "")
    if isinstance(gt, list): gt = gt[0]
    return str(gt).strip()


def generate_messages(current_input, few_shots):
    system_p = system_prompt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    messages = [{"role": "system", "content": system_p}]

    for fs in few_shots:
        fs_input = fs["input"].encode("utf-8", "ignore").decode("utf-8", "ignore")
        fs_output = format_few_shot_assistant(fs)
        messages.append({"role": "user", "content": f"Instruction:\n{fs_input}"})
        messages.append({"role": "assistant", "content": fs_output})

    text = current_input.encode("utf-8", "ignore").decode("utf-8", "ignore")
    messages.append(
        {"role": "user", "content": f"Instruction:\n{text}\n\nGenerate the code now (NO markdown, NO explanations):"})

    return messages


# ================= 清洗与代码有效性检查 =================

def remove_think(text: str) -> str:
    """去除推理模型的 <think> 标签及其内容"""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    elif "<think>" in text:
        return ""
    return text.strip()


def remove_markdown_code_block(text: str) -> str:
    """去除可能存在的 markdown 代码包裹"""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def check_code_validity(code: str) -> tuple[bool, str]:
    """
    尝试执行 Python 代码，返回是否成功执行及错误信息。
    融合了 check_code.py 的逻辑。
    """
    s = code.strip()
    if not s:
        return False, "Code is empty"

    # 基础关键词拦截（极速过滤明显非代码输出）
    if ("import " not in s) and ("@triton" not in s) and ("def " not in s):
        return False, "Missing essential keywords (import/def/@triton)"

    # 去掉@triton.jit装饰器，避免因 triton 装饰器导致的运行时异常
    code_to_eval = code.replace('@triton.jit', '#@triton.jit')

    try:
        # 使用空的 globals 命名空间执行，防止污染当前脚本环境
        exec(code_to_eval, {})
        return True, ""
    except Exception as e:
        # 捕获异常，仅返回异常类名和内容，不打印大量 traceback 以免刷屏
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, error_msg


# ================= 核心异步推理流 =================

async def process_single_sample(sample, few_shots, semaphore):
    async with semaphore:
        messages = generate_messages(sample["input"], few_shots)
        ground_truth = sample.get("output", None)

        for attempt, temp in enumerate(TEMPERATURE_STEPS):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temp,
                    max_tokens=8192,
                )

                raw_content = response.choices[0].message.content

                # ==== 检测 <think> 过程是否完整 ====
                has_start_think = "<think>" in raw_content
                has_end_think = "</think>" in raw_content

                if has_start_think and not has_end_think:
                    if attempt < MAX_RETRIES:
                        tqdm.write(
                            f"[Retry] 样本 {sample.get('id')} 思考过程未闭合(被截断)，重试 (Temp={TEMPERATURE_STEPS[attempt + 1]})...")
                        limit_instruction = "\n\n[CRITICAL REQUIREMENT]: 你的思考过程（<think>思考过程</think>）必须限制在100个字以内。降低思考复杂度。 Skip detailed analysis and output the final code immediately."
                        if limit_instruction not in messages[-1]["content"]:
                            messages[-1]["content"] += limit_instruction
                        continue
                    else:
                        tqdm.write(f"[Error] 样本 {sample.get('id')} {MAX_RETRIES} 次重试后思考过程仍未闭合。")
                # ====================================

                # 清理输出格式
                clean_content = remove_think(raw_content)
                clean_content = remove_markdown_code_block(clean_content)

                # ==== 严格的代码有效性检查 (Exec Check) ====
                is_valid, error_msg = check_code_validity(clean_content)

                if is_valid:
                    return {
                        "test_sample_id": sample.get("id", "unknown"),
                        "input": sample["input"],
                        "prediction": clean_content,
                        "ground_truth": ground_truth,
                        "retries_used": attempt,
                        "is_valid": True,
                        "error_log": ""
                    }
                else:
                    if attempt < MAX_RETRIES:
                        tqdm.write(
                            f"[Retry] 样本 {sample.get('id')} 代码执行校验失败 [{error_msg}]，反馈错误并重试 (Temp={TEMPERATURE_STEPS[attempt + 1]})...")

                        # ================= 新增：将错误信息反馈给模型 =================
                        # 1. 记录模型本次生成的错误代码 (作为 history)
                        messages.append({"role": "assistant", "content": raw_content})
                        # 2. 以 User 身份抛出报错信息，要求修复
                        feedback_prompt = (
                            f"Your previous code execution failed with the following error:\n"
                            f"```\n{error_msg}\n```\n"
                            f"Please fix the error and output the complete, corrected code. "
                            f"Output ONLY the final code. DO NOT include markdown formatting or explanations."
                        )
                        messages.append({"role": "user", "content": feedback_prompt})
                        # ==============================================================
                        continue
                    else:
                        tqdm.write(
                            f"[Error] 样本 {sample.get('id')} {MAX_RETRIES} 次重试后代码仍无效。最终报错: {error_msg}")
                        # 记录最后一次的错误
                        last_error_msg = error_msg

            except Exception as e:
                if attempt < MAX_RETRIES:
                    tqdm.write(f"[API Retry] 样本 {sample.get('id')} API调用失败: {e}，正在重试...")
                    await asyncio.sleep(2)
                    continue
                else:
                    tqdm.write(f"[Fatal] 样本 {sample.get('id')} API持续调用失败: {e}")
                    last_error_msg = f"API Error: {str(e)}"

        # 失败兜底返回
        return {
            "test_sample_id": sample.get("id", "unknown"),
            "input": sample["input"],
            "prediction": clean_content if 'clean_content' in locals() else DEFAULT_VALUE,  # 保留最后生成的错误代码用于分析
            "ground_truth": ground_truth,
            "retries_used": MAX_RETRIES,
            "is_valid": False,
            "error_log": last_error_msg if 'last_error_msg' in locals() else "Unknown failure"
        }


# ================= 主函数 =================

async def main():
    random.seed(42)

    if not os.path.exists(DATA_FILE):
        print(f"数据文件不存在: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    # 1. 数据划分
    knowledge_pool = task_data.get("examples", [])
    test_samples = task_data.get("examples", [])  # 根据原代码逻辑这里用的是examples做测试

    if not test_samples:
        print("未找到测试数据，请检查数据集结构。")
        return

    # 调试阶段抽样
    if len(test_samples) > 170:
        test_samples = random.sample(test_samples, 2)
    else:
        test_samples = test_samples

    print(f"任务：Triton Kernel Generation (With Exec Validation & Error Feedback)")
    print(f"检索池大小 (examples): {len(knowledge_pool)}")
    print(f"当前推理样本数 (test_samples): {len(test_samples)}")
    print(f"模型: {MODEL_NAME} | 并发: {CONCURRENCY_LIMIT}")

    # 2. 为每个样本预计算动态 Few-Shot
    print("正在为每个样本计算动态 Few-shot (Semantic Overlap)...")
    tasks_with_few_shots = []
    for sample in tqdm(test_samples, desc="Pre-computing Few-Shots"):
        best_few_shots = retrieve_dynamic_few_shots(sample, knowledge_pool, k=FEW_SHOT_K)
        tasks_with_few_shots.append((sample, best_few_shots))

    # 3. 并发推理
    print("开始并发推理...\n")
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_single_sample(sample, fs, semaphore) for sample, fs in tasks_with_few_shots]
    results = await tqdm.gather(*tasks, desc="Inference Progress")

    # 4. 文件与结果保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    results_v1 = [{"test_sample_id": r["test_sample_id"], "prediction": r["prediction"]} for r in results]

    # 保存全量结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n")

    # 保存比赛/评测 V1 格式
    with open(OUTPUT_FILE_V1, "w", encoding="utf-8") as f:
        for r_v1 in results_v1:
            f.write(json.dumps(r_v1, ensure_ascii=False) + "\n")

    # 记录生成失败或非法的 Case
    errors = [r for r in results if not r.get("is_valid")]
    if errors:
        with open(ERROR_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False, indent=2) + "\n")
        print(f"\n⚠️ 发现 {len(errors)} 个异常样本，已记录至: {ERROR_OUTPUT_FILE}")

    # 简单统计有效代码生成率
    valid_count = sum(1 for r in results if r["is_valid"])
    print(
        f"\n🎯 格式与执行有效率 (Valid Generation Rate): {valid_count / len(results):.2%} ({valid_count}/{len(results)})")
    print(f"✅ 详细结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
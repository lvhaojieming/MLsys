#!/usr/bin/env python3
"""Re-run three existing benchmark questions with Qwen thinking enabled."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

MODELS = {
    "gptq": "/home/zjh/mlsys/model/Qwen3-14B-GPTQ-Int4",
    "awq": "/home/zjh/mlsys/model/Qwen3-14B-AWQ",
}
SELECTED = [
    ("baseline_gsm8k", 89),
    ("baseline_mmlu_high_school_mathematics_generative", 3),
    ("baseline_mmlu_high_school_mathematics_generative", 12),
]
ROOT = Path("/home/zjh/mlsys/test-acc")
DEFAULT_OUT = Path("dataset/quantization_comparison/thinking_three_20261001")


def examples():
    wanted = set(SELECTED)
    found = {}
    for line in (ROOT / "full_gptq_20260930/all_samples.jsonl").open():
        row = json.loads(line)
        key = (row["task"], row["doc_id"])
        if key in wanted:
            found[key] = row
    assert set(found) == wanted
    return [found[key] for key in SELECTED]


def messages(row):
    question = row["doc"]["question"]
    if "choices" in row["doc"]:
        question += "\n" + "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(row["doc"]["choices"]))
        instruction = "Solve the problem carefully. Give your final answer as one uppercase letter: A, B, C, or D."
    else:
        instruction = "Solve the problem carefully. End your final answer with '#### ' followed by the numeric answer."
    return [{"role": "system", "content": instruction}, {"role": "user", "content": question}]


def split_answer(raw):
    if "</think>" not in raw:
        return raw.removeprefix("<think>\n"), "", False
    reasoning, answer = raw.split("</think>", 1)
    return reasoning.removeprefix("<think>\n").strip(), answer.strip(), True


def explicit_answer(row):
    text = row["final_answer"]
    if "choices" in row["doc"]:
        matches = re.findall(r"\\boxed\{([ABCD])\}", text)
        return matches[-1] if matches else None
    matches = re.findall(r"####\s*([-+]?\d[\d,.]*)", text)
    if not matches:
        matches = re.findall(r"\\boxed\{([-+]?\d[\d,.]*)\}", text)
    return matches[-1] if matches else None


def report(output):
    runs = {name: json.loads((output / f"{name}.json").read_text()) for name in MODELS}
    assert runs["gptq"]["generation"] == runs["awq"]["generation"]
    assert all(len(run["examples"]) == len(SELECTED) for run in runs.values())
    lines = ["# GPTQ / AWQ：开启 thinking 的三题完整输出", "", "两个 expert 使用同一个 AWQ tokenizer/chat template、相同问题与提示、相同生成参数。", "", "本次明确允许思考，MMLU 不再使用原来禁止推理的系统提示；这些是新的运行结果。推理文本为模型实际生成内容，未补写。", "", "生成参数：`" + json.dumps(runs["gptq"]["generation"], ensure_ascii=False) + "`", "", "仅比较两个 INT4 checkpoint，未运行未量化基座。", ""]
    lines.extend([f"计算 dtype：GPTQ `{runs['gptq']['dtype']}`，AWQ `{runs['awq']['dtype']}`（沿用各 checkpoint 的 auto 加载）。该对照不单独隔离量化算法与计算精度差异。", ""])
    lines.extend(["| 题目 | 标准答案 | GPTQ thinking | AWQ thinking |", "|---|---|---|---|"])
    for g, a in zip(runs["gptq"]["examples"], runs["awq"]["examples"]):
        target = g["target"] if "choices" in g["doc"] else g["target"].rsplit("####", 1)[-1].strip()
        lines.append(f"| {g['task']} / {g['doc_id']} | {target} | {explicit_answer(g)} | {explicit_answer(a)} |")
    lines.append("")
    paired = []
    for i, (g, a) in enumerate(zip(runs["gptq"]["examples"], runs["awq"]["examples"]), 1):
        assert (g["task"], g["doc_id"], g["prompt_sha256"]) == (a["task"], a["doc_id"], a["prompt_sha256"])
        assert g["input_ids"] == a["input_ids"]
        lines.extend([f"## 题目 {i}：{g['task']} / {g['doc_id']}", "", g["messages"][-1]["content"], "", "标准答案：", "", "````text", str(g["target"]), "````", ""])
        for name, row in (("GPTQ INT4", g), ("AWQ INT4", a)):
            lines.extend([f"### {name}", "", f"生成 {row['generated_tokens']} tokens；耗时 {row['seconds']:.2f}s；思考结束标记：{row['thinking_closed']}；达到长度上限：{row['hit_length_limit']}。", "", "完整 assistant 输出：", "", "````text", row["assistant_text"], "````", ""])
        paired.append({"task": g["task"], "doc_id": g["doc_id"], "question": g["doc"], "target": g["target"], "parsed_final_answers": {"gptq": explicit_answer(g), "awq": explicit_answer(a)}, "gptq": g, "awq": a})
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "paired.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in paired:
        print(json.dumps({"task": row["task"], "doc_id": row["doc_id"], "target": row["target"], **{k: {field: row[k][field] for field in ("final_answer", "generated_tokens", "thinking_closed", "hit_length_limit")} for k in MODELS}}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(output)
        return
    if args.quant is None:
        parser.error("--quant is required unless --report-only")
    destination = output / f"{args.quant}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODELS["awq"], local_files_only=True)
    if args.quant == "awq":
        import transformers.activations as activations
        if not hasattr(activations, "PytorchGELUTanh"):
            activations.PytorchGELUTanh = activations.GELUTanh
    print(f"LOADING {args.quant}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODELS[args.quant], device_map="cuda:0", dtype="auto", local_files_only=True)
    model.eval()
    print(f"LOADED {args.quant} dtype={model.dtype}", flush=True)
    settings = {"enable_thinking": True, "do_sample": False, "max_new_tokens": 4096, "seed": 42, "tokenizer": MODELS["awq"]}
    result = {"model_path": MODELS[args.quant], "dtype": str(model.dtype), "generation": settings, "examples": []}
    for row in examples():
        prompt_messages = messages(row)
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        assert prompt.endswith(("<|im_start|>assistant\n", "<think>\n")), repr(prompt[-100:])
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        torch.cuda.synchronize()
        start = time.monotonic()
        print(f"GENERATING {args.quant} {row['task']} {row['doc_id']}", flush=True)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=settings["max_new_tokens"], pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        # Some Qwen templates leave the opening think tag for the model to emit.
        # Only reconstruct that prefix when it actually appears in the prompt.
        assistant = ("<think>\n" if prompt.endswith("<think>\n") else "") + raw
        reasoning, answer, closed = split_answer(assistant)
        record = {"task": row["task"], "doc_id": row["doc_id"], "doc": row["doc"], "target": row["target"], "messages": prompt_messages, "prompt": prompt,
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_ids": inputs["input_ids"][0].tolist(), "generated_ids": ids,
                  "generated_tokens": len(ids), "generated_text": raw, "assistant_text": assistant, "reasoning_text": reasoning, "final_answer": answer,
                  "thinking_closed": closed, "hit_length_limit": len(ids) >= settings["max_new_tokens"] and ids[-1] != tokenizer.eos_token_id, "seconds": elapsed}
        result["examples"].append(record)
        partial = output / f"{args.quant}.partial.json"
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"DONE {args.quant} {row['doc_id']} tokens={len(ids)} thinking_closed={closed} answer={answer!r}", flush=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()

# gsm8k_groq_vs_agent.py
# Evaluate Groq (LangChain) vs your async agent on GSM8K CSV.
# Produces:
#   - outputs/gsm8k_eval.csv   (combined report)
#   - outputs/accuracy.png     (bar chart)
#   - outputs/groq_answers.csv (Groq-only answers)
#   - outputs/agent_answers.csv (Agent-only answers)

import os
import re
import math
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, List

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LangChain + Groq
from langchain_groq import ChatGroq

# Your async agent
from client import run_agent


# --------------------------
# SIMPLE CONFIG VARIABLES
# --------------------------
CSV_PATH = "outputs/main_test.csv"   # <-- put your CSV path here
NUM_ROWS = 15                        # <-- how many rows to evaluate
OUTDIR   = "outputs"                 # <-- where to write report files
GROQ_MODEL = "llama-3.1-8b-instant"  # <-- Groq model name
# --------------------------


SYSTEM_INSTRUCTION = (
    "You are a careful math solver. Show brief reasoning and ensure the final line "
    "contains only the numeric answer. If the result is an integer, print it as an integer."
)


def extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip()
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", s)
    if m:
        return m.group(1)
    m = re.search(r"(?i)(?:answer\s*[:=]\s*)([-+]?\d+(?:\.\d+)?)", s)
    if m:
        return m.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
    if nums:
        return nums[-1]
    return None


def answers_match(pred: Optional[str], truth: Optional[str], *, atol=1e-8, rtol=1e-9) -> bool:
    if pred is None or truth is None:
        return False
    try:
        return math.isclose(float(pred), float(truth), rel_tol=rtol, abs_tol=atol)
    except ValueError:
        na = re.sub(r"\s+", "", str(pred)).lower()
        nb = re.sub(r"\s+", "", str(truth)).lower()
        return na == nb


def build_prompt(question: str) -> str:
    return f"{SYSTEM_INSTRUCTION}\n\nQuestion:\n{question}\n\nFinal line must be the numeric answer only."


def make_groq_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=api_key)


async def call_agent_batch(questions: List[str]) -> List[str]:
    outputs = []
    for q in questions:
        tid = str(uuid.uuid4())
        try:
            resp = await run_agent(q, thread_id=tid)
        except Exception as e:
            resp = f"[AGENT_ERROR] {e}"
        outputs.append(str(resp))
    return outputs


def call_groq_batch(llm: ChatGroq, questions: List[str]) -> List[str]:
    outputs = []
    for q in questions:
        prompt = build_prompt(q)
        try:
            ai_msg = llm.invoke(prompt)
            resp = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
        except Exception as e:
            resp = f"[GROQ_ERROR] {e}"
        outputs.append(resp)
    return outputs


def save_model_answers(df: pd.DataFrame, outdir: Path):
    """Save Groq answers and Agent answers into separate files."""
    groq_df = df[["idx", "question", "true_answer_raw", "groq_raw", "groq_extracted", "groq_correct"]]
    agent_df = df[["idx", "question", "true_answer_raw", "agent_raw", "agent_extracted", "agent_correct"]]

    groq_path = outdir / "groq_answers.csv"
    agent_path = outdir / "agent_answers.csv"
    groq_df.to_csv(groq_path, index=False, encoding="utf-8")
    agent_df.to_csv(agent_path, index=False, encoding="utf-8")
    print(f"[✓] Saved Groq answers -> {groq_path}")
    print(f"[✓] Saved Agent answers -> {agent_path}")


def make_report(df_eval: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "gsm8k_eval.csv"
    df_eval.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[✓] Saved evaluation sheet -> {csv_path}")

    # Accuracies
    g_acc = df_eval["groq_correct"].mean() if len(df_eval) else 0.0
    a_acc = df_eval["agent_correct"].mean() if len(df_eval) else 0.0

    # Bar chart
    fig = plt.figure(figsize=(7, 4.5))
    labels = ["Groq (LangChain)", "My Agent"]
    vals = [g_acc, a_acc]
    plt.bar(labels, vals)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(f"GSM8K Accuracy (first {len(df_eval)})")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.0%}", ha="center")
    plt.tight_layout()
    img_path = outdir / "accuracy.png"
    fig.savefig(img_path, dpi=160)
    print(f"[✓] Saved accuracy chart -> {img_path}")


def main():
    df = pd.read_csv(CSV_PATH, engine="python")
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain columns: 'question' and 'answer'")

    k = min(NUM_ROWS, len(df))
    # randomly select k rows instead of taking the first k
    dfk = df.sample(n=k).copy()  # remove random_state if you want different set each run

    questions = dfk["question"].astype(str).tolist()
    gold_raw = dfk["answer"].astype(str).tolist()
    gold_extracted = [extract_final_answer(x) for x in gold_raw]

    # Groq
    llm = make_groq_llm()
    groq_raw = call_groq_batch(llm, questions)
    groq_extracted = [extract_final_answer(x) for x in groq_raw]
    groq_correct = [answers_match(p, t) for p, t in zip(groq_extracted, gold_extracted)]

    # Agent
    agent_raw = asyncio.run(call_agent_batch(questions))
    agent_extracted = [extract_final_answer(x) for x in agent_raw]
    agent_correct = [answers_match(p, t) for p, t in zip(agent_extracted, gold_extracted)]

    eval_df = pd.DataFrame({
        "idx": range(1, k + 1),
        "question": questions,
        "true_answer_raw": gold_raw,
        "true_answer_extracted": gold_extracted,
        "groq_raw": groq_raw,
        "groq_extracted": groq_extracted,
        "groq_correct": groq_correct,
        "agent_raw": agent_raw,
        "agent_extracted": agent_extracted,
        "agent_correct": agent_correct,
    })

    outdir = Path(OUTDIR)
    make_report(eval_df, outdir)
    save_model_answers(eval_df, outdir)

    summary = {
        "n": k,
        "groq_accuracy": round(float(eval_df["groq_correct"].mean()), 4),
        "agent_accuracy": round(float(eval_df["agent_correct"].mean()), 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

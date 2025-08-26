# table_matplotlib.py
# Build a color-coded table comparing Agent vs Groq answers.
# - Green cell: model's extracted answer is correct
# - Red cell:   model's extracted answer is wrong (shows true answer)
# Output: outputs/answers_table.png

import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------- config ----------
AGENT_CSV = "outputs/agent_answers.csv"
GROQ_CSV  = "outputs/groq_answers.csv"
OUT_PNG   = "outputs/answers_table.png"

# ---------- helpers ----------
def extract_final_answer(text: str):
    """Extract the final numeric-looking answer from a freeform solution string."""
    if text is None:
        return None
    s = str(text).strip()
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", s)
    if m:
        return m.group(1)
    m = re.search(r"(?i)(?:answer\s*[:=]\s*)([-+]?\d+(?:\.\d+)?)", s)
    if m:
        return m.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else None

def answers_match(pred, truth, *, atol=1e-8, rtol=1e-9) -> bool:
    """Numeric compare with fallback to string equality."""
    if pred is None or truth is None:
        return False
    try:
        ap = float(pred); tr = float(truth)
        return abs(ap - tr) <= max(atol, rtol * abs(tr))
    except Exception:
        return str(pred).strip().lower() == str(truth).strip().lower()

def pick_merge_keys(agent_cols, groq_cols):
    """Choose the strongest common merge keys present in both CSVs."""
    for keys in (["idx", "question", "true_answer_raw"],
                 ["idx", "true_answer_raw"],
                 ["idx"]):
        if set(keys).issubset(agent_cols) and set(keys).issubset(groq_cols):
            return keys
    if "idx" in agent_cols and "idx" in groq_cols:
        return ["idx"]
    raise KeyError("No common merge keys found; ensure both CSVs share at least 'idx'.")

# ---------- load ----------
agent_df = pd.read_csv(AGENT_CSV)
groq_df  = pd.read_csv(GROQ_CSV)

# Ensure 'idx' exists (fallback numbering if missing)
if "idx" not in agent_df.columns:
    agent_df = agent_df.copy()
    agent_df["idx"] = range(1, len(agent_df) + 1)
if "idx" not in groq_df.columns:
    groq_df = groq_df.copy()
    groq_df["idx"] = range(1, len(groq_df) + 1)

# Pick merge keys dynamically
keys = pick_merge_keys(agent_df.columns, groq_df.columns)

# Keep only needed columns if present
agent_keep = [c for c in ["idx","question","true_answer_raw","agent_raw","agent_extracted","agent_correct"] if c in agent_df.columns]
groq_keep  = [c for c in ["idx","question","true_answer_raw","groq_raw","groq_extracted","groq_correct"] if c in groq_df.columns]

merged = pd.merge(
    agent_df[agent_keep],
    groq_df[groq_keep],
    on=keys,
    how="inner",
    suffixes=("_agent","_groq")
)

# Determine/compute true answer (extracted)
if "true_answer_raw" in merged.columns:
    merged["true_extracted"] = merged["true_answer_raw"].apply(extract_final_answer)
else:
    cand = [c for c in merged.columns if c.endswith("true_answer_raw")]
    merged["true_extracted"] = merged[cand[0]].apply(extract_final_answer) if cand else None

# Ensure extracted predictions exist
if "agent_extracted" not in merged.columns and "agent_raw" in merged.columns:
    merged["agent_extracted"] = merged["agent_raw"].apply(extract_final_answer)
if "groq_extracted" not in merged.columns and "groq_raw" in merged.columns:
    merged["groq_extracted"]  = merged["groq_raw"].apply(extract_final_answer)

# Compute correctness if missing
if "agent_correct" not in merged.columns:
    merged["agent_correct"] = [
        answers_match(p, t) for p, t in zip(merged.get("agent_extracted"), merged.get("true_extracted"))
    ]
if "groq_correct" not in merged.columns:
    merged["groq_correct"] = [
        answers_match(p, t) for p, t in zip(merged.get("groq_extracted"), merged.get("true_extracted"))
    ]

# ---------- friendly numbering: 1..N ----------
merged = merged.copy()
merged["Q#"] = range(1, len(merged) + 1)

# ---------- build matplotlib table ----------
table_rows = [["Q#", "Agent Answer", "Groq Answer"]]
cell_colors = [["#f5f5f5", "#f5f5f5", "#f5f5f5"]]

for _, row in merged.iterrows():
    q = int(row["Q#"])
    truth = row.get("true_extracted")

    # Agent cell
    a_pred = row.get("agent_extracted")
    a_ok   = bool(row.get("agent_correct", False))
    if a_ok:
        a_text  = str(a_pred)
        a_color = "#c8e6c9"  # light green
    else:
        a_text  = f"{a_pred} (true: {truth})"
        a_color = "#ffcdd2"  # light red

    # Groq cell
    g_pred = row.get("groq_extracted")
    g_ok   = bool(row.get("groq_correct", False))
    if g_ok:
        g_text  = str(g_pred)
        g_color = "#c8e6c9"
    else:
        g_text  = f"{g_pred} (true: {truth})"
        g_color = "#ffcdd2"

    table_rows.append([q, a_text, g_text])
    cell_colors.append(["#ffffff", a_color, g_color])

# Plot height scales with row count (min height 4 inches)
fig_h = max(4, len(table_rows) * 0.4)
fig, ax = plt.subplots(figsize=(12, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText=table_rows,
    cellColours=cell_colors,
    loc="center",
    cellLoc="left",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.3)

plt.tight_layout()
Path(os.path.dirname(OUT_PNG) or ".").mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=160)
print(f"[✓] Saved: {OUT_PNG}")

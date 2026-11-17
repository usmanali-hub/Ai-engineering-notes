"""Print the tables used in the README."""

from sampling_lab import LOGITS, TOKENS, order_matters, sample_counts, softmax, temperature_table, top_k_filter, top_p_filter, summarize


def table(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


print("## Temperature (no top-p)\n")
print(table(temperature_table([0.0, 0.3, 0.7, 1.0, 1.5, 2.0])))

print("\n## Top-p at temperature 1.0 vs a fixed top-k of 5: tokens still possible\n")
rows = []
for name, logits in LOGITS.items():
    base = softmax(logits, 1.0)
    row = {"situation": name}
    for p in (0.5, 0.8, 0.9, 0.95):
        row[f"top_p {p}"] = summarize(top_p_filter(base, p))["tokens possible"]
    row["top_k 5"] = summarize(top_k_filter(base, 5))["tokens possible"]
    rows.append(row)
print(table(rows))

print("\n## 10,000 draws, confident situation, temperature 2.0 with and without top-p 0.9\n")
logits = LOGITS["confident (capital of France is ...)"]
for label, probs in (("T=2.0 only", softmax(logits, 2.0)), ("T=2.0, top_p 0.9", top_p_filter(softmax(logits, 2.0), 0.9))):
    counts = sample_counts(probs, 10_000)
    shown = ", ".join(f"{TOKENS[i].strip()}={counts[i]}" for i in counts.argsort()[::-1][:6])
    print(f"- {label}: {shown}; distinct tokens drawn: {(counts > 0).sum()}")

print("\n## Does the order of temperature and top-p matter? (ambiguous situation, T=2.0, top_p 0.9)\n")
a, b = order_matters(LOGITS["ambiguous (two good answers)"], 2.0, 0.9)
print("temperature first:", [round(float(x), 3) for x in a[:6]])
print("top-p first:      ", [round(float(x), 3) for x in b[:6]])

#!/usr/bin/env python3
# A2_extract_train_loss.py
import json, os, sys, csv
from glob import glob

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/shivraj-pg/DEPNECT/OUT_gemma4B_conllu"

ckpts = sorted(
    glob(os.path.join(OUT_DIR, "checkpoint-*")),
    key=lambda x: int(os.path.basename(x).split("-")[-1]),
)

rows = []

for ck in ckpts:
    trainer_state_path = os.path.join(ck, "trainer_state.json")
    if not os.path.isfile(trainer_state_path):
        continue

    with open(trainer_state_path, "r", encoding="utf-8") as f:
        js = json.load(f)

    log_history = js.get("log_history", [])

    for entry in log_history:
        # training loss entries have "loss"
        if "loss" in entry:
            rows.append({
                "checkpoint": os.path.basename(ck),
                "path": ck,
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "train_loss": entry.get("loss")
            })

out_csv = os.path.join(OUT_DIR, "train_loss_trace.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["checkpoint", "path", "step", "epoch", "train_loss"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("Wrote:", out_csv)
print("First 20 lines:")
for r in rows[:20]:
    print(r)

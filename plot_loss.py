import pandas as pd
import matplotlib.pyplot as plt
import sys, os

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/shivraj-pg/DEPNECT/OUT_gemma4B_conllu"
csv_path = os.path.join(OUT_DIR, "train_loss_trace.csv")

df = pd.read_csv(csv_path)
df = df.sort_values(by="step")

plt.figure(figsize=(8,5))
plt.plot(df["step"], df["train_loss"])
plt.xlabel("Training steps")
plt.ylabel("Training loss")
plt.title("Training Loss Curve")
out_path = os.path.join(OUT_DIR, "train_loss_plot.png")
plt.savefig(out_path, bbox_inches="tight")

print("Saved:", out_path)

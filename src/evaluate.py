# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

metrics = pd.read_csv("/Users/nsumesh/Documents/GitHub/641HW3/src/results/metrics.csv")
metrics.columns = [c.strip() for c in metrics.columns]

print(f"Loaded {len(metrics)} experiments from metrics.csv")

plt.figure(figsize=(8, 5))
sns.lineplot(data=metrics, x="Seq Length", y="Accuracy", hue="Model", marker="o", errorbar=None)
plt.title("Accuracy vs Sequence Length")
plt.tight_layout()
plt.savefig("results/accuracy_vs_seq_len.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(data=metrics, x="Seq Length", y="F1 Score", hue="Model", marker="o", errorbar=None)
plt.title("F1 Score vs Sequence Length")
plt.tight_layout()
plt.savefig("results/f1_vs_seq_len.png", dpi=300)
plt.show()

print("Saved plots: accuracy_vs_seq_len.png and f1_vs_seq_len.png")

best_loss_file = "/Users/nsumesh/Documents/GitHub/641HW3/src/results/best_model.csv"
worst_loss_file = "/Users/nsumesh/Documents/GitHub/641HW3/src/results/worst_model.csv"

best_loss = pd.read_csv(best_loss_file)
plt.figure(figsize=(7, 4))
plt.plot(best_loss["loss"], label="Training Loss")
plt.plot(best_loss["val_loss"], label="Validation Loss")
plt.title("Best Model: LSTM (Adam, Seq=100)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/best_model_loss_curve.png", dpi=300)
plt.show()

worst_loss = pd.read_csv(worst_loss_file)
plt.figure(figsize=(7, 4))
plt.plot(worst_loss["loss"], label="Training Loss")
plt.plot(worst_loss["val_loss"], label="Validation Loss")
plt.title("Worst Model: RNN (SGD, Seq=25)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/worst_model_loss_curve.png", dpi=300)
plt.show()

print("Saved best_model_loss_curve.png and worst_model_loss_curve.png")

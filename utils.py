import os
import matplotlib.pyplot as plt

def plot_metrics(train_loss, val_loss, save_path):
os.makedirs(save_path, exist_ok=True) # Ensure the plot directory exists

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Train Loss', marker='o')
plt.plot(val_loss, label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Triplet Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(save_path, "training_loss.png")
plt.savefig(plot_path)
plt.close()

print(f"✅ Loss plot saved at: {plot_path}")

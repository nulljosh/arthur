import matplotlib.pyplot as plt
import json
from datetime import datetime

# Parse training logs
with open("logs/training.log", "r") as f:
    losses = [float(line.split("loss: ")[1].split()[0]) for line in f if "loss:" in line]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Arthur v2 Training Progress")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("docs/training_curve.png")
print(f"📊 Saved training curve to docs/training_curve.png")

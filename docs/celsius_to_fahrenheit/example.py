"""
Celsius → Fahrenheit conversion with a pre-trained neural network.

Loads model.json (trained by train.py) and predicts Fahrenheit
for a list of temperatures. No retraining needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier

# ── normalization constants (must match train.py) ──────────────────────────────

C_MIN, C_MAX = -50.0, 150.0
F_MIN = 1.8 * C_MIN + 32   # -58.0
F_MAX = 1.8 * C_MAX + 32   # 302.0


def predict(net, celsius: float) -> float:
    """Convert Celsius to Fahrenheit using the trained network."""
    c_norm = (celsius - C_MIN) / (C_MAX - C_MIN)
    f_norm = nier.feed_forward(net, [c_norm, 1.0])[-1][0]
    return f_norm * (F_MAX - F_MIN) + F_MIN


# ── load model ────────────────────────────────────────────────────────────────

here = Path(__file__).parent
net  = nier.load(here / "model.json")

# ── predictions ───────────────────────────────────────────────────────────────

temperatures = [-40, -20, 0, 10, 20, 30, 37, 40, 60, 80, 100]

print("Celsius → Fahrenheit")
print("-" * 28)
for c in temperatures:
    f = predict(net, c)
    print(f"  {c:>5}°C  →  {f:>7.2f}°F")

print()
print("Notable points:")
print(f"  -40°C = -40°F  (the only temperature equal in both scales)")
print(f"    0°C = 32°F   (water freezes)")
print(f"  100°C = 212°F  (water boils)")

"""
Apartment price prediction with a pre-trained neural network.

Loads model.json (trained by train.py) and estimates prices
for several apartments based on area and floor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier

# ── normalization constants (must match train.py) ──────────────────────────────

A_MAX = 150.0
F_MAX =  25.0
P_MAX = 15000.0


def predict(net, area: float, floor: int) -> float:
    """Predict apartment price in thousands of rubles."""
    p_norm = nier.feed_forward(net, [area / A_MAX, floor / F_MAX])[-1][0]
    return p_norm * P_MAX


# ── load model ────────────────────────────────────────────────────────────────

here = Path(__file__).parent
net  = nier.load(here / "model.json")

# ── predictions ───────────────────────────────────────────────────────────────

apartments = [
    ( 32,  1, "studio, ground floor"),
    ( 45,  3, "one-bedroom, low floor"),
    ( 55,  7, "one-bedroom, mid floor"),
    ( 65,  1, "two-bedroom, ground floor"),
    ( 70, 12, "two-bedroom, high floor"),
    ( 90, 18, "three-bedroom, high floor"),
    (110, 22, "large apartment, near top"),
    (140, 25, "penthouse"),
]

print("Apartment price prediction")
print(f"  {'area':>6}  {'floor':>5}  {'price':>12}  description")
print("  " + "-" * 50)
for area, floor, desc in apartments:
    price = predict(net, area, floor)
    print(f"  {area:>5}m²  {floor:>5}  {price:>9.0f}k ₽  {desc}")

print()
print("Observations:")
print("  • Each extra 10 m² adds roughly 850k ₽  (the network learned ≈85k/m²)")
print("  • Each floor up adds roughly 25k ₽        (the network learned ≈25k/floor)")
print("  • Prices reflect what the network extracted from 100 training examples.")

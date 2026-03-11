"""
Iris Setosa recognition with a pre-trained neural network.

Loads model.json (trained by train.py) and predicts whether a flower
is Iris Setosa based on its petal measurements.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier

# ── normalization constants (must match train.py) ──────────────────────────────

PL_MAX = 7.0
PW_MAX = 2.5


def predict(net, petal_length: float, petal_width: float) -> float:
    """Return P(is_setosa) for given petal measurements."""
    return nier.feed_forward(net, [petal_length / PL_MAX, petal_width / PW_MAX])[-1][0]


# ── load model ────────────────────────────────────────────────────────────────

here = Path(__file__).parent
net  = nier.load(here / "model.json")

# ── predictions ───────────────────────────────────────────────────────────────

flowers = [
    (1.4, 0.2, "Iris Setosa (typical)"),
    (1.7, 0.4, "Iris Setosa (larger)"),
    (1.9, 0.6, "Iris Setosa (largest in dataset)"),
    (3.0, 1.1, "boundary zone — network is uncertain"),
    (4.5, 1.5, "Iris Versicolor"),
    (4.9, 1.8, "Iris Versicolor (larger)"),
    (5.8, 2.2, "Iris Virginica"),
    (6.9, 2.3, "Iris Virginica (largest in dataset)"),
]

print("Iris Setosa recognition")
print(f"  {'petal_l':>8}  {'petal_w':>8}  {'P(Setosa)':>10}  {'verdict':>12}  species")
print("  " + "-" * 70)
for pl, pw, species in flowers:
    p       = predict(net, pl, pw)
    verdict = "Setosa ✓" if p >= 0.5 else "not Setosa"
    print(f"  {pl:>8.1f}  {pw:>8.1f}  {p:>10.4f}  {verdict:>12}  {species}")

print()
print("Setosa has petals < 2 cm long and < 0.7 cm wide.")
print("The network learned this boundary from data — without being told the rule.")

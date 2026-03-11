"""
Train a neural network to recognise Iris Setosa flowers.

Dataset: Fisher's Iris dataset (1936) — public domain.
Source:  https://archive.ics.uci.edu/dataset/53/iris

Features:
  petal_length  — length of the petal in cm
  petal_width   — width  of the petal in cm

Label:
  1 = Iris Setosa
  0 = other species (Versicolor or Virginica)

Setosa is completely separable from other species by petal size alone —
a perfect case for a Sigmoid classifier.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier
from nier.models import Activation, Layer

# ── Fisher's Iris dataset (petal_length, petal_width, species) ────────────────
# 50 Setosa + 50 Versicolor + 50 Virginica = 150 samples
# species: 0 = Setosa, 1 = Versicolor, 2 = Virginica

IRIS_DATA = [
    # Setosa (50 samples)
    (1.4, 0.2, 0), (1.4, 0.2, 0), (1.3, 0.2, 0), (1.5, 0.2, 0), (1.4, 0.2, 0),
    (1.7, 0.4, 0), (1.4, 0.3, 0), (1.5, 0.2, 0), (1.4, 0.2, 0), (1.5, 0.1, 0),
    (1.5, 0.2, 0), (1.6, 0.2, 0), (1.4, 0.1, 0), (1.1, 0.1, 0), (1.2, 0.2, 0),
    (1.5, 0.4, 0), (1.3, 0.4, 0), (1.4, 0.3, 0), (1.7, 0.3, 0), (1.5, 0.3, 0),
    (1.7, 0.2, 0), (1.5, 0.4, 0), (1.0, 0.2, 0), (1.7, 0.5, 0), (1.9, 0.2, 0),
    (1.6, 0.2, 0), (1.6, 0.4, 0), (1.5, 0.2, 0), (1.4, 0.2, 0), (1.6, 0.2, 0),
    (1.6, 0.2, 0), (1.5, 0.4, 0), (1.5, 0.1, 0), (1.4, 0.2, 0), (1.5, 0.2, 0),
    (1.2, 0.2, 0), (1.3, 0.2, 0), (1.4, 0.1, 0), (1.3, 0.2, 0), (1.5, 0.3, 0),
    (1.3, 0.3, 0), (1.3, 0.2, 0), (1.3, 0.6, 0), (1.6, 0.4, 0), (1.9, 0.3, 0),
    (1.4, 0.2, 0), (1.6, 0.2, 0), (1.4, 0.2, 0), (1.5, 0.4, 0), (1.4, 0.2, 0),
    # Versicolor (50 samples)
    (4.7, 1.4, 1), (4.5, 1.5, 1), (4.9, 1.5, 1), (4.0, 1.3, 1), (4.6, 1.5, 1),
    (4.5, 1.3, 1), (4.7, 1.6, 1), (3.3, 1.0, 1), (4.6, 1.3, 1), (3.9, 1.4, 1),
    (3.5, 1.0, 1), (4.2, 1.5, 1), (4.0, 1.0, 1), (4.7, 1.4, 1), (3.6, 1.3, 1),
    (4.4, 1.4, 1), (4.5, 1.5, 1), (4.1, 1.0, 1), (4.5, 1.5, 1), (3.9, 1.1, 1),
    (4.8, 1.8, 1), (4.0, 1.3, 1), (4.9, 1.5, 1), (4.7, 1.2, 1), (4.3, 1.3, 1),
    (4.4, 1.4, 1), (4.8, 1.4, 1), (5.0, 1.7, 1), (4.5, 1.5, 1), (3.5, 1.0, 1),
    (3.8, 1.1, 1), (3.7, 1.0, 1), (3.9, 1.2, 1), (5.1, 1.6, 1), (4.5, 1.5, 1),
    (4.5, 1.6, 1), (4.7, 1.5, 1), (4.4, 1.3, 1), (4.1, 1.3, 1), (4.0, 1.3, 1),
    (4.4, 1.2, 1), (4.6, 1.4, 1), (4.0, 1.2, 1), (3.3, 1.0, 1), (4.2, 1.3, 1),
    (4.2, 1.2, 1), (4.2, 1.3, 1), (4.3, 1.3, 1), (3.0, 1.1, 1), (4.1, 1.3, 1),
    # Virginica (50 samples)
    (6.0, 2.5, 2), (5.1, 1.9, 2), (5.9, 2.1, 2), (5.6, 1.8, 2), (5.8, 2.2, 2),
    (6.6, 2.1, 2), (4.5, 1.7, 2), (6.3, 1.8, 2), (5.8, 1.8, 2), (6.1, 2.5, 2),
    (5.1, 2.0, 2), (5.3, 1.9, 2), (5.5, 2.1, 2), (5.0, 2.0, 2), (5.1, 2.4, 2),
    (5.3, 2.3, 2), (5.5, 1.8, 2), (6.7, 2.2, 2), (6.9, 2.3, 2), (5.0, 1.5, 2),
    (5.7, 2.3, 2), (4.9, 2.0, 2), (6.7, 2.0, 2), (4.9, 1.8, 2), (5.7, 2.1, 2),
    (6.0, 2.4, 2), (4.8, 2.3, 2), (4.9, 1.9, 2), (5.6, 2.3, 2), (5.8, 2.5, 2),
    (6.1, 2.3, 2), (6.4, 1.9, 2), (5.6, 2.0, 2), (5.1, 2.3, 2), (5.6, 1.8, 2),
    (6.1, 2.2, 2), (5.6, 2.1, 2), (5.5, 2.1, 2), (4.8, 1.7, 2), (5.4, 1.8, 2),
    (5.6, 1.8, 2), (5.1, 1.8, 2), (5.9, 2.1, 2), (5.7, 1.6, 2), (5.2, 1.9, 2),
    (5.0, 2.0, 2), (5.2, 2.2, 2), (5.4, 1.5, 2), (5.1, 1.4, 2), (5.1, 2.3, 2),
]

# ── normalization constants ────────────────────────────────────────────────────

PL_MAX = 7.0   # petal_length max (rounded up from dataset max 6.9)
PW_MAX = 2.5   # petal_width  max (dataset max 2.5)


def normalize(petal_length, petal_width):
    return [petal_length / PL_MAX, petal_width / PW_MAX]


# ── helpers ───────────────────────────────────────────────────────────────────

def save_csv(data, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["petal_length", "petal_width", "is_setosa"])
        for pl, pw, species in data:
            writer.writerow([pl, pw, 1 if species == 0 else 0])


# ── training ──────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent

    save_csv(IRIS_DATA, here / "data.csv")
    print(f"Saved {len(IRIS_DATA)} samples → data.csv")
    print(f"  Setosa: {sum(1 for _,_,s in IRIS_DATA if s==0)}, "
          f"Others: {sum(1 for _,_,s in IRIS_DATA if s!=0)}")

    # Data order in IRIS_DATA is all-Setosa first — train() shuffles each epoch.
    X = [normalize(pl, pw) for pl, pw, _ in IRIS_DATA]
    y = [[1.0 if species == 0 else 0.0] for _, _, species in IRIS_DATA]

    np.random.seed(42)
    net = nier.create([
        Layer(2, Activation.SIGMOID),   # input: petal_length, petal_width
        Layer(4, Activation.SIGMOID),   # hidden
        Layer(1, Activation.SIGMOID),   # output: P(is_setosa)
    ])

    result = nier.train(net, X, y, epochs=300, alfa=0.1)

    first, last = result.loss_history[0], result.loss_history[-1]
    print(f"Loss:  {first:.4f} → {last:.6f}")

    # ── accuracy ──────────────────────────────────────────────────────────────
    trained = result.perceptron
    correct = 0
    for pl, pw, species in IRIS_DATA:
        prob     = nier.feed_forward(trained, normalize(pl, pw))[-1][0]
        predicted = 1 if prob >= 0.5 else 0
        label     = 1 if species == 0 else 0
        if predicted == label:
            correct += 1

    accuracy = correct / len(IRIS_DATA) * 100
    print(f"Train accuracy: {correct}/{len(IRIS_DATA)} ({accuracy:.1f}%)")

    # ── hand-crafted test examples ─────────────────────────────────────────────
    examples = [
        (1.4, 0.2, "typical Setosa"),
        (1.9, 0.4, "large Setosa"),
        (4.5, 1.5, "typical Versicolor"),
        (5.8, 2.2, "typical Virginica"),
        (2.5, 0.8, "ambiguous (between species)"),
    ]

    print("\n── Validation ──────────────────────────────────────────────────")
    print(f"  {'petal_l':>8}  {'petal_w':>8}  {'P(Setosa)':>10}  {'verdict':>12}  note")
    print("  " + "-" * 65)
    for pl, pw, note in examples:
        prob    = nier.feed_forward(trained, normalize(pl, pw))[-1][0]
        verdict = "Setosa" if prob >= 0.5 else "not Setosa"
        print(f"  {pl:>8.1f}  {pw:>8.1f}  {prob:>10.4f}  {verdict:>12}  {note}")

    nier.save(trained, here / "model.json")
    print("\nModel saved → model.json")


if __name__ == "__main__":
    main()

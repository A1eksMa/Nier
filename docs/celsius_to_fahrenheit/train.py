"""
Train a neural network to convert Celsius to Fahrenheit.

Runs training, prints a validation table, and saves the trained model
to model.json so example.py can load it without retraining.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier
from nier.models import Activation, Layer

# ── constants ─────────────────────────────────────────────────────────────────

C_MIN, C_MAX = -50.0, 150.0
F_MIN = 1.8 * C_MIN + 32   # -58.0
F_MAX = 1.8 * C_MAX + 32   # 302.0


# ── normalization helpers ──────────────────────────────────────────────────────

def norm_c(c):
    return (c - C_MIN) / (C_MAX - C_MIN)

def norm_f(f):
    return (f - F_MIN) / (F_MAX - F_MIN)

def denorm_f(f_norm):
    return f_norm * (F_MAX - F_MIN) + F_MIN


# ── data generation ───────────────────────────────────────────────────────────

def generate_data():
    """Generate (celsius, fahrenheit) pairs, C from -50 to 150 step 5."""
    return [(float(c), round(1.8 * c + 32, 2)) for c in range(-50, 151, 5)]


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["celsius", "fahrenheit"])
        writer.writerows(rows)


# ── training ──────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent

    rows = generate_data()
    save_csv(rows, here / "data.csv")
    print(f"Generated {len(rows)} samples → data.csv")

    # Normalize to [0, 1] for stable training.
    # Second input is always 1.0 — a "bias neuron" that lets the network
    # learn an offset (without it, output is forced through zero).
    X = [[norm_c(c), 1.0] for c, _ in rows]
    y = [[norm_f(f)] for _, f in rows]

    np.random.seed(0)
    net = nier.create([
        Layer(2, Activation.LINEAR),   # 2 inputs: C_norm and bias
        Layer(1, Activation.LINEAR),   # 1 output: F_norm
    ])

    result = nier.train(net, X, y, epochs=3000, alfa=0.01)

    first, last = result.loss_history[0], result.loss_history[-1]
    print(f"Loss:  {first:.6f} → {last:.8f}")

    # ── validation ────────────────────────────────────────────────────────────
    trained = result.perceptron
    test_cases = [
        (-40,  -40.0,  "freezing point (C = F)"),
        (  0,   32.0,  "water freezing"),
        ( 20,   68.0,  "room temperature"),
        ( 37,   98.6,  "body temperature"),
        (100,  212.0,  "water boiling"),
    ]

    print("\n── Validation ───────────────────────────────────────────────")
    print(f"  {'C':>5}  {'predicted F':>12}  {'expected F':>11}  {'error':>8}  note")
    print("  " + "-" * 60)
    for c, expected, note in test_cases:
        f_norm = nier.feed_forward(trained, [norm_c(c), 1.0])[-1][0]
        predicted = denorm_f(f_norm)
        error = abs(predicted - expected)
        print(f"  {c:>5}  {predicted:>12.4f}  {expected:>11.1f}  {error:>8.4f}  {note}")

    nier.save(trained, here / "model.json")
    print("\nModel saved → model.json")


if __name__ == "__main__":
    main()

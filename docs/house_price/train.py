"""
Train a neural network to predict apartment prices.

Features:
  area   — total area in square metres (30–150 m²)
  floor  — floor number (1–25)

Target:
  price  — price in thousands of rubles

The price follows an approximate linear rule with random noise:
  price ≈ 85 × area + 25 × floor + 500 (+ noise ±200k)

Runs training, prints validation examples, and saves the model to model.json.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier
from nier.models import Activation, Layer

# ── normalization constants ────────────────────────────────────────────────────

A_MAX = 150.0    # max area, m²
F_MAX =  25.0    # max floor
P_MAX = 15000.0  # max price, thousands of rubles


def normalize(area, floor):
    return [area / A_MAX, floor / F_MAX]

def denormalize_price(p_norm):
    return p_norm * P_MAX


# ── data generation ───────────────────────────────────────────────────────────

def generate_data(n=100, seed=42):
    """
    Synthetic apartment market data.
    Price formula: 85 × area + 25 × floor + 500 + noise(σ=200)
    """
    rng = np.random.default_rng(seed)
    area  = rng.uniform(30, 150, n)
    floor = rng.integers(1, 26, n).astype(float)
    price = 85 * area + 25 * floor + 500 + rng.normal(0, 200, n)
    price = np.clip(price, 0, None)
    return list(zip(area.tolist(), floor.tolist(), price.tolist()))


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["area_m2", "floor", "price_thousands_rub"])
        for area, floor, price in rows:
            writer.writerow([round(area, 1), int(floor), round(price, 0)])


# ── training ──────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent

    rows = generate_data()
    save_csv(rows, here / "data.csv")
    print(f"Generated {len(rows)} samples → data.csv")
    prices = [p for _, _, p in rows]
    print(f"Price range: {min(prices):.0f}k – {max(prices):.0f}k rubles")

    X = [normalize(area, floor) for area, floor, _ in rows]
    y = [[price / P_MAX]        for _,    _,    price in rows]

    np.random.seed(42)
    net = nier.create([
        Layer(2, Activation.LINEAR),   # input: area, floor
        Layer(6, Activation.RELU),     # hidden: ReLU extracts nonlinear patterns
        Layer(1, Activation.LINEAR),   # output: price (unbounded — no saturation)
    ])

    result = nier.train(net, X, y, epochs=1000, alfa=0.05)

    first, last = result.loss_history[0], result.loss_history[-1]
    print(f"Loss:  {first:.5f} → {last:.6f}")

    # ── error on training data ─────────────────────────────────────────────────
    trained = result.perceptron
    errors = []
    for i, (area, floor, price_true) in enumerate(rows):
        price_pred = denormalize_price(nier.feed_forward(trained, X[i])[-1][0])
        errors.append(abs(price_pred - price_true))

    rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
    print(f"Mean absolute error: {sum(errors)/len(errors):.0f}k rubles")
    print(f"RMSE:                {rmse:.0f}k rubles")

    # ── hand-crafted validation examples ──────────────────────────────────────
    examples = [
        ( 40,  2, "small studio, low floor"),
        ( 55,  5, "one-bedroom, mid floor"),
        ( 75, 10, "two-bedroom, high floor"),
        (100, 15, "three-bedroom, high floor"),
        (130, 20, "large apartment, very high floor"),
    ]

    print("\n── Validation ──────────────────────────────────────────────────")
    print(f"  {'area':>6}  {'floor':>5}  {'predicted':>12}  {'formula':>10}  description")
    print("  " + "-" * 65)
    for area, floor, desc in examples:
        pred    = denormalize_price(nier.feed_forward(trained, normalize(area, floor))[-1][0])
        formula = 85 * area + 25 * floor + 500
        print(f"  {area:>6}m²  {floor:>5}  {pred:>9.0f}k ₽  {formula:>8.0f}k ₽  {desc}")

    nier.save(trained, here / "model.json")
    print("\nModel saved → model.json")


if __name__ == "__main__":
    main()

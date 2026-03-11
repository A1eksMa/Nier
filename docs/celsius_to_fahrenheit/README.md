# Case: Celsius → Fahrenheit

**Activation:** Linear
**Task type:** Regression (learning an exact mathematical formula)

---

## What is this?

We teach a neural network to convert temperature from Celsius to Fahrenheit.

The formula is:

```
F = 1.8 × C + 32
```

The network does not know this formula. It looks at examples — pairs of
(°C, °F) — and figures out the rule on its own.

---

## Why Linear activation?

The relationship between Celsius and Fahrenheit is **linear**: the graph is a
straight line. A network with Linear activation in every layer can only learn
straight-line relationships, which makes it the perfect and exact tool for
this task.

Using Sigmoid or ReLU here would be unnecessary — they are designed for
curved, nonlinear relationships.

---

## The bias neuron

There is a subtlety: the formula `F = 1.8 × C + 32` has an **offset** (+32).
Without a special trick, a neuron can only learn `F = w × C` (a line through
zero), which is wrong.

The solution is a **bias neuron**: we always pass `1.0` as a second input
alongside `C`. The network then learns two weights:

```
F = w₁ × C + w₂ × 1.0
  = 1.8 × C + 32
```

After training, the network learns `w₁ ≈ 1.8` and `w₂ ≈ 32`.

---

## Architecture

```
Input layer:   2 neurons  (C_normalized, 1.0)   — Linear
Output layer:  1 neuron   (F_normalized)         — Linear
```

Inputs and outputs are normalized to the range [0, 1] for stable training.
The `predict()` function in `example.py` handles normalization/denormalization
transparently.

---

## Files

| File | Description |
|---|---|
| `data.csv` | 41 training samples: °C from −50 to +150, step 5 |
| `train.py` | Trains the network and saves `model.json` |
| `example.py` | Loads `model.json` and prints predictions |
| `model.json` | Pre-trained model (ready to use) |

---

## Quick start

Run the example immediately (no training needed):

```bash
python docs/celsius_to_fahrenheit/example.py
```

To retrain from scratch:

```bash
python docs/celsius_to_fahrenheit/train.py
```

---

## Expected output

```
Celsius → Fahrenheit
----------------------------
    -40°C  →   -40.00°F
      0°C  →    32.00°F
     37°C  →    98.60°F
    100°C  →   212.00°F
```

The network learns the formula with essentially **zero error** — because the
task is perfectly linear and the network architecture matches the task exactly.

---

## What this demonstrates

- A network with **Linear activation** = a linear function. Nothing more,
  nothing less.
- This is both a strength (exact fit for linear data) and a limitation
  (cannot learn curved relationships — see the ReLU and Sigmoid examples).
- The **bias neuron** is a universal technique used in all neural networks
  to allow learning offsets.

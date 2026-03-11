# Case: Apartment Price Prediction

**Activation:** ReLU (hidden layers) + Linear (output)
**Task type:** Regression (predicting a continuous value)

---

## What is this?

We teach a neural network to estimate apartment prices.

The network receives two inputs:
- `area` — total area in square metres (30–150 m²)
- `floor` — floor number (1–25)

And it must output:
- `price` — estimated price in thousands of rubles

---

## The data

100 synthetic apartments generated from a realistic pricing formula:

```
price ≈ 85 × area + 25 × floor + 500  (+ random noise ±200k)
```

This means: every extra square metre adds ~85k ₽, every floor up adds ~25k ₽,
and there is a base price of 500k ₽. Random noise models the real-world
variation that the formula alone cannot capture.

---

## Why ReLU + Linear?

### ReLU in hidden layers

ReLU (Rectified Linear Unit) passes positive values unchanged and sets
negative values to zero: `f(x) = max(0, x)`.

A combination of many ReLU neurons can approximate **any shape of curve**,
not just straight lines. This makes the network flexible enough to handle
cases where the relationship between inputs and price is not perfectly linear
(e.g. high floor + large area could have a premium not captured by simple addition).

### Linear output

The output layer must produce an **unbounded number** — price can be any
positive value. Using Sigmoid or ReLU on the output would clip the range
to (0, 1) or (0, ∞) in a distorted way.

Linear activation on the output means: "output = whatever the network
calculates" — no squashing, no clipping.

**Rule of thumb:**
- Regression → Linear output
- Probability / yes-no → Sigmoid output

---

## Architecture

```
Input layer:   2 neurons  (area / 150,  floor / 25)    — Linear (unused)
Hidden layer:  6 neurons                               — ReLU
Output layer:  1 neuron   (price / 15000)              — Linear
```

Inputs and output are normalised so that all numbers stay near 1,
which helps the optimizer work consistently.

---

## Files

| File | Description |
|---|---|
| `data.csv` | 100 synthetic apartments: area, floor, price |
| `train.py` | Trains the network (1000 epochs) and saves `model.json` |
| `example.py` | Loads `model.json` and predicts prices for 8 apartments |
| `model.json` | Pre-trained model (ready to use) |

---

## Quick start

Run the example immediately (no training needed):

```bash
python docs/house_price/example.py
```

To retrain from scratch:

```bash
python docs/house_price/train.py
```

---

## Expected output

```
Apartment price prediction
    area  floor         price  description
  --------------------------------------------------
     32m²      1       2913k ₽  studio, ground floor
     55m²      7       5106k ₽  one-bedroom, mid floor
     90m²     18       8513k ₽  three-bedroom, high floor
    140m²     25      13131k ₽  penthouse
```

---

## What this demonstrates

- **ReLU hidden layers** are the standard building block for regression —
  they introduce nonlinearity and can approximate any continuous function.
- **Linear output** is the correct choice whenever the target is an
  unbounded real number (price, temperature, distance, time…).
- The network learns the pricing coefficients (≈85k/m², ≈25k/floor)
  purely from examples — without being given the formula.
- **Mean absolute error ≈ 175k ₽** on prices ranging from 3 400k to 13 500k —
  roughly 2–3% relative error, comparable to the noise in the training data.

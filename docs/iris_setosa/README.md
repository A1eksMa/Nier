# Case: Iris Setosa Recognition

**Activation:** Sigmoid
**Task type:** Binary classification

---

## What is this?

We teach a neural network to identify **Iris Setosa** — one of three species
of iris flower — by the size of its petals.

The network is given two measurements:
- `petal_length` — length of the petal in centimetres
- `petal_width` — width of the petal in centimetres

And it must answer: **is this flower Iris Setosa?** (yes / no)

---

## Dataset

This example uses the **Fisher Iris dataset** (Ronald Fisher, 1936) —
one of the most famous datasets in machine learning, used in textbooks
for over 80 years. It is in the public domain.

Original source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)

The dataset contains 150 flower measurements:

| Species | Samples | Petal length | Petal width |
|---|---|---|---|
| Iris Setosa | 50 | 1.0 – 1.9 cm | 0.1 – 0.6 cm |
| Iris Versicolor | 50 | 3.0 – 5.1 cm | 1.0 – 1.8 cm |
| Iris Virginica | 50 | 4.5 – 6.9 cm | 1.4 – 2.5 cm |

**Setosa is completely separated from the other species** — there is no
overlap in petal measurements. This makes it a perfect case for
a Sigmoid classifier.

---

## Why Sigmoid activation?

The network must output a **probability**: how likely is it that this
flower is Setosa?

Sigmoid maps any number to a value between 0 and 1, which is exactly
what we need:

```
P(Setosa) = 0.97  →  almost certainly Setosa
P(Setosa) = 0.03  →  almost certainly not Setosa
P(Setosa) = 0.50  →  uncertain (near the boundary)
```

The decision threshold is 0.5: above it we say "Setosa", below — "not Setosa".

---

## Architecture

```
Input layer:   2 neurons  (petal_length / 7.0,  petal_width / 2.5)  — Sigmoid
Hidden layer:  4 neurons                                              — Sigmoid
Output layer:  1 neuron   (P(is_setosa))                             — Sigmoid
```

Inputs are normalised by the maximum values in the dataset so all
numbers stay in a similar range, which helps training.

---

## Files

| File | Description |
|---|---|
| `data.csv` | 150 real samples from the Fisher Iris dataset |
| `train.py` | Trains the network (300 epochs) and saves `model.json` |
| `example.py` | Loads `model.json` and classifies several flowers |
| `model.json` | Pre-trained model (ready to use) |

---

## Quick start

Run the example immediately (no training needed):

```bash
python docs/iris_setosa/example.py
```

To retrain from scratch:

```bash
python docs/iris_setosa/train.py
```

---

## Expected output

```
Iris Setosa recognition
   petal_l   petal_w   P(Setosa)       verdict  species
  ----------------------------------------------------------------------
       1.4       0.2      0.9674      Setosa ✓  Iris Setosa (typical)
       3.0       1.1      0.0618    not Setosa  boundary zone
       4.5       1.5      0.0084    not Setosa  Iris Versicolor
       5.8       2.2      0.0025    not Setosa  Iris Virginica
```

The network achieves **100% accuracy** on the full dataset —
because Setosa is perfectly separable by petal size.

---

## What this demonstrates

- **Sigmoid activation** is the natural choice when you need probabilities
  between 0 and 1 — that is, for any yes/no question.
- The network learns a **decision boundary** in 2D space that separates
  Setosa from the other species, without being told where that boundary is.
- **Per-epoch shuffling** (enabled by default in `nier.train`) is essential
  with imbalanced datasets: here 50 Setosa vs 100 others. Without shuffling,
  the training order biases the gradients and the network fails to converge.

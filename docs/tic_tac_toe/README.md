# Case: Tic-tac-toe — Did X Win?

**Activation:** Sigmoid
**Task type:** Binary classification (pattern recognition)

---

## What is this?

We teach a neural network to look at a tic-tac-toe board and answer:
**has X won the game?**

The board is a 3×3 grid. Each cell contains:
- `1` — X is here
- `-1` — O is here
- `0` — empty

The network receives all 9 cell values and outputs a single probability.

```
X | X | X        P(X wins) = 0.992   →   X WINS ✓
---+---+---
. | O | .
---+---+---
. | O | .
```

```
X | X | .        P(X wins) = 0.004   →   no winner
---+---+---
O | O | .
---+---+---
. | . | .
```

---

## The rules (that the network must discover on its own)

X wins if it occupies any complete **line**:
- 3 rows (top / middle / bottom)
- 3 columns (left / centre / right)
- 2 diagonals (main / anti)

**The network is never told these rules.** It must infer them from 234 labelled
board examples.

---

## Why Sigmoid?

This is a yes/no question: "did X win?" Sigmoid maps the network's output
to a probability between 0 and 1:

```
P = 0.99  →  almost certainly yes
P = 0.01  →  almost certainly no
P = 0.50  →  the network is uncertain
```

The threshold is 0.5: above — X wins, below — no winner.

---

## Architecture

```
Input layer:   9 neurons  (one per board cell: 1, -1, or 0)  — Sigmoid
Hidden layer: 12 neurons  (detect line patterns)             — Sigmoid
Output layer:  1 neuron   (P(X has won))                     — Sigmoid
```

12 hidden neurons is enough to represent all 8 winning lines plus
the context of surrounding cells.

---

## Files

| File | Description |
|---|---|
| `data.csv` | 234 board states (117 X-wins, 117 no-wins), balanced |
| `train.py` | Trains the network (2000 epochs) and saves `model.json` |
| `example.py` | Loads `model.json` and analyses 6 board positions |
| `model.json` | Pre-trained model (ready to use) |

---

## Quick start

Run the example immediately (no training needed):

```bash
python docs/tic_tac_toe/example.py
```

To retrain from scratch:

```bash
python docs/tic_tac_toe/train.py
```

---

## Expected output

```
====================================================
  Tic-tac-toe: did X win?
====================================================

Top row — X wins  —  P(X wins) = 0.992  [X WINS ✓]
  X | X | X
  ---+---+---
  . | O | .
  ---+---+---
  . | O | .

Draw — board full, no winner  —  P(X wins) = 0.030  [no winner]
  X | X | O
  ---+---+---
  O | O | X
  ---+---+---
  X | O | X
```

---

## What this demonstrates

- **Sigmoid** is the right activation for any classification problem — it
  gives a probability and a clear threshold decision.
- The network learns **structural patterns** (lines in a grid), not just
  statistical correlations. This is pattern recognition in its purest form.
- With only 234 examples and 12 hidden neurons, the network achieves
  **99.6% accuracy** — generalising to board positions it has never seen.
- This case is more complex than the previous ones: the input has 9 dimensions
  instead of 2, and the relevant structure (a complete line) involves
  combining exactly 3 specific cells out of 9.

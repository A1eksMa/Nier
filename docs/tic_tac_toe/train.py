"""
Train a neural network to recognise whether X has won a tic-tac-toe game.

Input:  9 values representing the board (1 = X, -1 = O, 0 = empty)
Output: P(X has won) — a value between 0 and 1

The network must learn to detect 8 winning patterns (3 rows, 3 columns,
2 diagonals) purely from examples, without being given any rules.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier
from nier.models import Activation, Layer


# ── game logic ────────────────────────────────────────────────────────────────

WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def x_wins(board):
    return any(board[a] == board[b] == board[c] == 1 for a, b, c in WINNING_LINES)

def o_wins(board):
    return any(board[a] == board[b] == board[c] == -1 for a, b, c in WINNING_LINES)


# ── data generation ───────────────────────────────────────────────────────────

def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    boards, labels = [], []

    # ── X-win boards ──────────────────────────────────────────────────────────
    for a, b, c in WINNING_LINES:
        base = [0] * 9
        base[a] = base[b] = base[c] = 1
        empty = [i for i in range(9) if base[i] == 0]

        for _ in range(15):
            board = base.copy()
            n_o = int(rng.integers(1, 4))
            o_cells = rng.choice(empty, size=min(n_o, len(empty)), replace=False)
            for i in o_cells:
                board[i] = -1
            if x_wins(board) and not o_wins(board):
                boards.append(board)
                labels.append(1.0)

    # ── non-win boards ────────────────────────────────────────────────────────
    attempts = 0
    while len([l for l in labels if l == 0.0]) < len([l for l in labels if l == 1.0]):
        board = [0] * 9
        cells = rng.permutation(9)
        n_x = int(rng.integers(1, 5))
        n_o = int(rng.integers(1, 5))
        for i in cells[:n_x]:
            board[i] = 1
        for i in cells[n_x:n_x + n_o]:
            board[i] = -1
        if not x_wins(board) and not o_wins(board):
            boards.append(board)
            labels.append(0.0)
        attempts += 1
        if attempts > 10000:
            break

    return boards, labels


def save_csv(boards, labels, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["c1","c2","c3","c4","c5","c6","c7","c8","c9","x_wins"])
        for board, label in zip(boards, labels):
            writer.writerow(board + [int(label)])


# ── training ──────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent

    boards, labels = generate_data()
    save_csv(boards, labels, here / "data.csv")

    n_wins    = sum(1 for l in labels if l == 1.0)
    n_no_wins = sum(1 for l in labels if l == 0.0)
    print(f"Generated {len(labels)} boards: {n_wins} X-wins, {n_no_wins} no-wins → data.csv")

    X = [board for board in boards]
    y = [[label] for label in labels]

    np.random.seed(0)
    net = nier.create([
        Layer(9,  Activation.SIGMOID),   # input: 9 board cells
        Layer(12, Activation.SIGMOID),   # hidden: learns to detect line patterns
        Layer(1,  Activation.SIGMOID),   # output: P(X has won)
    ])

    result = nier.train(net, X, y, epochs=2000, alfa=0.1)

    first, last = result.loss_history[0], result.loss_history[-1]
    print(f"Loss:  {first:.4f} → {last:.6f}")

    trained = result.perceptron
    correct = sum(
        1 for i in range(len(labels))
        if (nier.feed_forward(trained, X[i])[-1][0] >= 0.5) == (labels[i] == 1.0)
    )
    print(f"Train accuracy: {correct}/{len(labels)} ({correct/len(labels)*100:.1f}%)")

    # ── validation on specific boards ─────────────────────────────────────────
    test_cases = [
        ([1, 1, 1,  0,-1, 0,  0,-1, 0],  True,  "X wins: top row"),
        ([1, 0, 0,  1, 0, 0,  1, 0, 0],  True,  "X wins: left column"),
        ([1, 0, 0,  0, 1, 0,  0, 0, 1],  True,  "X wins: main diagonal"),
        ([0, 0, 1,  0, 1,-1,  1,-1, 0],  True,  "X wins: anti-diagonal"),
        ([1, 1, 0, -1,-1, 1,  0, 1,-1],  False, "no winner yet"),
        ([1,-1, 1, -1, 1,-1, -1, 1,-1],  False, "draw — board full, no winner"),
        ([0, 0, 0,  0, 0, 0,  0, 0, 0],  False, "empty board"),
        ([-1,-1,-1, 1, 0, 1,  0, 1, 0],  False, "O wins (not X)"),
    ]

    print("\n── Validation ──────────────────────────────────────────────────────")
    def fmt_board(b):
        sym = {1:"X", -1:"O", 0:"."}
        return " ".join(sym[v] for v in b[:3]) + " | " + \
               " ".join(sym[v] for v in b[3:6]) + " | " + \
               " ".join(sym[v] for v in b[6:])

    for board, expected, note in test_cases:
        p = nier.feed_forward(trained, board)[-1][0]
        verdict = "X WINS" if p >= 0.5 else "no win"
        ok = "✓" if (p >= 0.5) == expected else "✗"
        print(f"  [{fmt_board(board)}]  P={p:.3f}  {verdict}  {ok}  {note}")

    nier.save(trained, here / "model.json")
    print("\nModel saved → model.json")


if __name__ == "__main__":
    main()

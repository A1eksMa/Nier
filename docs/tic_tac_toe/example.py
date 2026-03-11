"""
Tic-tac-toe winner detection with a pre-trained neural network.

The network receives a board state (9 values) and outputs
the probability that X has won.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nier

# ── load model ────────────────────────────────────────────────────────────────

here = Path(__file__).parent
net  = nier.load(here / "model.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def predict(board):
    """Return P(X has won) for the given board."""
    return nier.feed_forward(net, board)[-1][0]


def draw_board(board):
    """Render a 3×3 board in ASCII."""
    sym = {1: "X", -1: "O", 0: "."}
    rows = [board[0:3], board[3:6], board[6:9]]
    lines = [" | ".join(sym[v] for v in row) for row in rows]
    separator = "\n  ---+---+---\n  "
    return "  " + separator.join(lines)


def analyse(board, description):
    p = predict(board)
    verdict = "X WINS ✓" if p >= 0.5 else ("no winner" if p < 0.1 else "uncertain")
    print(f"{description}  —  P(X wins) = {p:.3f}  [{verdict}]")
    print(draw_board(board))
    print()


# ── examples ──────────────────────────────────────────────────────────────────

print("=" * 52)
print("  Tic-tac-toe: did X win?")
print("=" * 52)
print()

analyse(
    [1, 1, 1,
     0,-1, 0,
     0,-1, 0],
    "Top row — X wins"
)

analyse(
    [1, 0, 0,
     0, 1, 0,
     0, 0, 1],
    "Main diagonal — X wins"
)

analyse(
    [0, 0, 1,
     0, 1,-1,
     1,-1, 0],
    "Anti-diagonal — X wins"
)

analyse(
    [1, 1,-1,
    -1,-1, 1,
     1,-1, 1],
    "Draw — board full, no winner"
)

analyse(
    [-1,-1,-1,
      1, 0, 1,
      0, 1, 0],
    "O wins (not X)"
)

analyse(
    [1, 1, 0,
    -1,-1, 0,
     0, 0, 0],
    "Game in progress — no winner yet"
)

print("The network learned to detect all 8 winning patterns")
print("(3 rows, 3 columns, 2 diagonals) from examples alone.")

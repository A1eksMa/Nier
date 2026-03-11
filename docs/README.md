# Nier — Examples

Each subdirectory is a self-contained example that demonstrates one
activation function through a real, runnable task.

Every example includes:
- A detailed `README.md` explaining the task, the architecture, and the concepts
- `data.csv` with the training data
- `train.py` to train the network and save a model
- `example.py` to load the pre-trained model and show predictions
- `model.json` — the pre-trained model, ready to use without retraining

---

## Examples

| Directory | Activation | Task |
|---|---|---|
| [`celsius_to_fahrenheit/`](celsius_to_fahrenheit/) | Linear | Learning the formula F = 1.8 × C + 32 |
| [`iris_setosa/`](iris_setosa/) | Sigmoid | Recognising Iris Setosa from petal measurements (Fisher dataset, 1936) |
| [`house_price/`](house_price/) | ReLU + Linear | Regression: predicting apartment price from area and floor |
| [`tic_tac_toe/`](tic_tac_toe/) | Sigmoid | Pattern recognition: did X win the tic-tac-toe game? |

---

## Running any example

From the project root:

```bash
python docs/<example_name>/example.py
```

To retrain:

```bash
python docs/<example_name>/train.py
```

# HIV Molecule Classification with GNN / CNN / SVM

HKUST **COMP 4222** (Machine Learning) group project: predict HIV-related molecular activity from graph-structured molecule data.

We compare three approaches on the same dataset:

1. **GNN** — Graph Transformer layers (PyTorch Geometric)
2. **CNN** — GraphSAGE embeddings + 1D CNN classifier
3. **SVM** — classical baseline on GraphSAGE features

---

## Problem

Molecular compounds are naturally **graphs** (atoms = nodes, bonds = edges).  
This project converts SMILES / molecular records into graphs, then trains models to classify them — a common pattern in cheminformatics and drug discovery ML.

---

## Highlights

| Model | Idea | Stack |
|-------|------|--------|
| **GNN** | End-to-end learning on molecular graphs | PyTorch Geometric `TransformerConv`, TopK pooling |
| **CNN** | GraphSAGE embeddings → 1D CNN | GraphSAGE + CNN |
| **SVM** | Classical baseline on fixed embeddings | scikit-learn SVM |

Dataset handling includes an **oversampled** training CSV to mitigate class imbalance.

---

## Project structure

```
Graph-Neural-Networks/
├── data/
│   └── raw/                 # HIV train (oversampled) + test CSV
├── GNN/                     # Full GNN train / eval pipeline
├── CNN/                     # GraphSAGE + CNN
├── SVM/                     # SVM baseline
└── requirements.txt
```

### What each folder does

**`GNN/`**
- `data_preprocessing.py` — molecule → PyG graph (cached under `data/processed`)
- `model.py` — Graph Transformer GNN
- `train.py` / `test.py` / `main.py` — training and evaluation entrypoints
- `hyper_parameters.py` — tunable hyperparameters
- `utils.py` — metrics / helpers

**`CNN/`**
- GraphSAGE embedding + 1D CNN classifier

**`SVM/`**
- Fit an SVM on GraphSAGE features

---

## Setup

```bash
git clone https://github.com/JerryHung1103/Graph-Neural-Networks.git
cd Graph-Neural-Networks

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

> Tip: PyTorch Geometric install can be platform-specific. If `pip install -r requirements.txt` fails on PyG packages, follow the official [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA / CPU setup, then retry.

---

## Usage

### Train GNN

```bash
cd GNN
python main.py
```

First run may take longer while molecular strings are converted to graphs.

### Train CNN

```bash
cd CNN
python train.py
```

GraphSAGE encoding before CNN training can take time.

### Fit SVM baseline

```bash
cd SVM
python train.py
```

---

## Skills demonstrated

- Graph neural networks for structured scientific data
- Comparing deep (GNN/CNN) vs classical (SVM) baselines
- Data preprocessing / imbalance handling
- Reproducible experiment layout (separate model folders + shared data)

---

## Course

COMP 4222 — Group 22 project (HKUST).

---

## License

Academic course project for portfolio / learning use.

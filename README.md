# COMP 4222 Group 22 Project
## 1. Project File Structure
- **data/**
  - **raw/**
    - README.md
    - HIV_train_oversampled.csv
    - test.csv

- **GNN/**
  - data_preprocessing.py
  - hyper_parameters.py
  - main.py
  - model.py
  - test.py
  - train.py
  - utils.py

- **CNN/**
  - featurizer.py
  - graph_sage_embedding.py
  - model.py
  - train.py
 
  ## 2. General Purpose of Each File

- **data/**
  - **raw/** Contains raw molecular data stored in CSV format.
    - `HIV_train_oversampled.csv`: Dataset with oversampled instances to address class imbalance.
    - `test.csv`: Dataset for testing and validation purposes.

- **GNN/**
  - `data_preprocessing.py`: Implements a graph converter that transforms molecular strings into graph-structured data using PyTorch Geometric, storing the processed data in `data/processed`.
  - `hyper_parameters.py`: Stores hyperparameters for the GNN model for easy tuning.
  - `main.py`: Main script for training and evaluating the GNN model.
  - `model.py`: Defines the architecture and implementation of the GNN model.
  - `test.py`: Provides testing functionality for evaluating the model performance.
  - `train.py`: Implements training functionality for the GNN model.
  - `utils.py`: Contains utility functions for evaluating the model's performance, such as calculating confusion metrics and determining the total number of model parameters.

- **CNN/**
  - `featurizer.py`: Provides functions for converting graph objects into embeddings using GraphSAGE.
  - `graph_sage_embedding.py`: Defines the architecture of GraphSAGE.
  - `model.py`: Defines the architecture of a 1-dimensional CNN.
  - `train.py`: Contains the main logic for training the CNN model.

- `requirements.txt`: Specifies the packages and their versions used in this project.

## 3. Usage

### Setting Up the Environment

1. **Optional:** Setting up a virtual environment:
    ```bash
    python -m venv .venv
    # Activate the virtual environment
    .venv\Scripts\activate
    ```

2. Installing required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Training GNN

1. Navigate to the GNN directory:
    ```bash
    cd GNN
    ```

2. Run the main script for training the GNN model:
    ```bash
    python main.py
    ```

    **Note:** For the first run, it may take some time to convert the molecular string into a graph.

### Training CNN

1. Navigate to the CNN directory:
    ```bash
    cd CNN
    ```

2. Run the training script for the CNN model:
    ```bash
    python train.py
    ```

    **Note:** Encoding the graph using GraphSAGE before training the CNN model may take some time.

---

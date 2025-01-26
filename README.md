# LeRobot-Federated-Robotics
Flower.ai and LeRobot (Hugging Face): Federated Learning for Scalable Robotics

# Federated Application with Flower and Hugging Face

This project is located in the main directory [lerobot_example](https://github.com/adap/flower/tree/main/examples/quickstart-lerobot/lerobot_example). This code implements a **federated application** using **Flower** and **Hugging Face** to train models with distributed data. It is part of a **research project** focused on exploring federated learning techniques in the context of real-world robotics. Below is a brief description of the structure and main parts of the implementation.

## 1. Client: `client_app.py`

The client implements the **training** and **evaluation** logic for the local model and is responsible for communicating with the server to update the global model.

### Main functionalities:
- **Initialization**: The client is initialized with a partition ID, a number of local epochs, a data loader for training, and the device (CPU/GPU).
- **Methods**:
  - **fit**: Trains the model using local data.
  - **evaluate**: Evaluates the trained model and returns loss and performance metrics.

## 2. Server `server_app.py`

The server manages the **aggregation strategy** for the models trained by the clients and updates the global model.

### Main functionalities:
- **Aggregation strategy**: **FedAvg** is used to combine local models into a global model.
- **Periodic evaluation**: Evaluations are set to occur every few rounds, and global model checkpoints are saved.
- **Functions**: Specific functions are used to configure evaluation and save the model state.

## 3. Task `task.py`

The task defines the dataset and its transformation. In this case, the **dataset `lerobot/pusht`** is used.

### Main functionalities:
- **Dataset**: The `lerobot/pusht` dataset and its corresponding transformations are defined.
- **Data partitioning**: `GroupedNaturalIdPartitioner` is used to partition the data to simulate multiple training nodes.

## 4. Flower Application

Flower is used to combine the **clients** and the **server** into distributed applications that allow federated model training.

### Main functionalities:
- **Federation**: Data is partitioned using `GroupedNaturalIdPartitioner` to simulate multiple training nodes in a federated system.
- **Diffusion Policy Model**: `DiffusionPolicy` is used to configure a specific environment for training.

## Implementation Summary

- **Model**: Utilizes diffusion policies (`DiffusionPolicy`) configured for the `pusht` environment.
- **Federation**: Federation is implemented through Flower, using `NumPyClient` on the clients and **FedAvg** as the aggregation strategy on the server.
- **Data**: The data is partitioned to simulate a federated environment with multiple training nodes.



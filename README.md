# Federated Application with Flower and LeRobot (Hugging Face)

This project is located in the main directory **[Flower&LeRobot_example](https://github.com/adap/flower/tree/main/examples/quickstart-lerobot/lerobot_example)**. This code implements a **federated application** using **[Flower](https://flower.ai)** and **[LeRobot from Hugging Face](https://github.com/huggingface/lerobot)** to train models with distributed data. It is part of a **research project** focused on exploring federated learning techniques in the context of real-world robotics.

![LeRobot Federated Robotics](https://github.com/albertovalverde/LeRobot-Federated-Robotics/blob/main/_static/render_compose.gif)


## Overview of the Original Repository: Flower & LeRobot Federated Learning

This experiment focuses on training a robot to correctly position letters of the alphabet. Currently, the model is trained to place the letter "T," with the goal of eventually extending it to the entire alphabet. The project is part of a broader research initiative aimed at exploring federated learning techniques within real-world robotics applications. To better understand the project, you can watch the video below explaining the experiment:
[Project Overview Video](https://www.youtube.com/watch?v=fwAtTOZttWo)


## Research on Federated Learning for Humanoid Robots

This research explores how **federated learning** can improve humanoid robots in **privacy-sensitive environments** like hospitals, homes, factories, or even industries. These robots, equipped with cameras and sensors, collect valuable interaction data that cannot be shared due to privacy concerns.

Federated learning enables these robots to enhance their models collaboratively across different clients without exchanging sensitive data. This approach ensures that the robots benefit from a global model while maintaining privacy, making it ideal for environments where security is paramount and data sharing is not feasible.

## Broader Applications of Federated Learning

Federated learning can be applied to any device, system, or application where privacy is crucial, including healthcare and personal spaces. **It is particularly promising for the legal field**, where privacy concerns are paramount but technological advancements are necessary. This approach allows robust model development without compromising sensitive data, offering a solution to balance privacy with progress in critical sectors.


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

## 4. Flower Framework

[Flower](https://flower.ai) is used to combine the **clients** and the **server** into distributed applications that allow federated model training.

### Main functionalities:
- **Federation**: Data is partitioned using `GroupedNaturalIdPartitioner` to simulate multiple training nodes in a federated system.
- **Diffusion Policy Model**: `DiffusionPolicy` is used to configure a specific environment for training.

## Implementation Summary

- **Model**: Utilizes diffusion policies (`DiffusionPolicy`) configured for the `pusht` environment.
- **Federation**: Federation is implemented through Flower, using `NumPyClient` on the clients and **FedAvg** as the aggregation strategy on the server.
- **Data**: The data is partitioned to simulate a federated environment with multiple training nodes.

  
## Set up the project

### Clone the project

Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone https://github.com/albertovalverde/LeRobot-Federated-Robotics.git

```

This will create a new directory called `quickstart-lerobot` containing the following files:

```shell
quickstart-lerobot
├── lerobot_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── task.py         # Defines your model, training and data loading
│   ├── lerobot_federated_dataset.py   # Defines the dataset
│   └── configs/		# configuration files
│ 		├── env/        	# gym environment config
│   	├── policy/			# policy config
│   	└── default.yaml 	# default config settings
│
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `lerobot_example` package.

```bash
pip install -e .
```

### Choose training parameters

You can leave the default parameters for an initial quick test. It will run for 50 rounds sampling 4 clients per round. However for best results, total number of training rounds should be at least 100,000. You can achieve this for example by setting `num-server-rounds=500` and `local_epochs=200` in the `pyproject.toml`.

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine. You can read more about how the Simulation Engine work [in the documentation](https://flower.ai/docs/framework/how-to-run-simulations.html).

### Run with the Simulation Engine

> \[!TIP\]
> This example runs much faster when the `ClientApp`s have access to a GPU. If your system has one, you might want to try running the example with GPU right away, use the `local-simulation-gpu` federation as shown below.

```bash
# Run with the default federation (CPU only)
flwr run .
```

Run the project in the `local-simulation-gpu` federation that gives CPU and GPU resources to each `ClientApp`. By default, at most 2x`ClientApp` (using ~2 GB of VRAM each) will run in parallel in each available GPU. Note you can adjust the degree of parallelism but modifying the `client-resources` specification. Running with the settings as in the `pyproject.toml` it takes 1h in a 2x RTX 3090 machine.

```bash
# Run with the `local-simulation-gpu` federation
flwr run . local-simulation-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run . local-simulation-gpu --run-config "num-server-rounds=5 fraction-fit=0.1"
```

### Result output

Results of training steps for each client and server logs will be under the `outputs/` directory. For each run there will be a subdirectory corresponding to the date and time of the run. For example:

```shell
outputs/date_time/
├── evaluate  # Each subdirectory contains .mp4 renders generated by clients
│   ├── round_5	# Evaluations in a given round
│	│   ├── client_3
│	│	...	└── rollout_20241207-105418.mp4 # render .mp4 for client at a given round
│	│	└── client_1
│   ...
│   └── round_n   	# local client model checkpoint
└── global_model # Each subdirectory contains the global model of a round
	├── round_1
	...
	└── round_n
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.




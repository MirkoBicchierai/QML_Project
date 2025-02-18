# Quantum Support Vector Data Description for Anomaly Detection

This repository contains implementations of the DSVDD, QAE, and QSVDD models from the paper "Quantum support vector data description for anomaly detection". The goal of this work is to showcase the efficiency of quantum methods for anomaly detection tasks, utilizing significantly fewer parameters compared to classical approaches.

## Features

- Implementations of DSVDD, QAE, and QSVDD models
- Tested on CIFAR10, KMNIST, MNIST, and FMNIST datasets
- Model saving and result storage
- Integration with COMET ML for experiment tracking

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: [list the required libraries]

### Installation

1. Clone the repository:
    ```
   git clone https://github.com/your-username/quantum-anomaly-detection.git
    ```
2. Install the required libraries:
    ```
   pip install -r requirements.txt
    ```
### Setup

1. Create a `Dataset` folder in the project directory to store the datasets.

## Usage

1. To run the DSVDD model:
   ```
   python SVDD.py
   ```
2. To run the QAE model:
    ```
   python QAE.py
    ```
3. To run the QSVDD model:
   ```
   python QSVDD.py
   ```
   
Each script will execute all the tests with the CIFAR10, KMNIST, MNIST, and FMNIST datasets.

## Model Saving and Results

- Trained models are saved in the `Models` directory.
- Results are stored in the `Results` folder.

## Experiment Tracking

All experiments are registered with COMET ML. You can view the experiment details and results at:
[COMET ML Link](https://www.comet.com/mirkobicchierai/qml/view/new/panels)

## Model Parameters

| Model  | Number of Parameters |
|--------|----------------------|
| DSVDD  | 92                   |
| QAE    | 78                   |
| QSVDD  | 75                   |

## Some Interesting result Results

| Dataset | Method | Condition | Mean AUC | Mean Accuracy |
|---------|--------|-----------|----------|---------------|
| MNIST   | QSVDD  | Noiseless | 78.79    | 75.57         |
|         |        | Noise     | 70.91    | 69.10         |
|         | QAE    | Noiseless | 74.77    | 71.80         |
|         |        | Noise     | 61.50    | 63.26         |
|         | DSVDD  | /         | 65.79    | 65.37         |
| FMNIST  | QSVDD  | Noiseless | 88.54    | 83.62         |
|         |        | Noise     | 74.99    | 72.10         |
|         | QAE    | Noiseless | 81.01    | 78.35         |
|         |        | Noise     | 75.17    | 74.77         |
|         | DSVDD  | /         | 75.66    | 72.23         |
| KMNIST  | QSVDD  | Noiseless | 66.16    | 64.44         |
|         |        | Noise     | 57.55    | 59.53         |
|         | QAE    | Noiseless | 60.59    | 62.25         |
|         |        | Noise     | 53.86    | 58.03         |
|         | DSVDD  | /         | 59.71    | 60.39         |
| CIFAR10 | QSVDD  | Noiseless | 59.03    | 59.74         |
|         |        | Noise     | 55.94    | 59.31         |
|         | QAE    | Noiseless | 56.94    | 55.19         |
|         |        | Noise     | 58.00    | 59.29         |
|         | DSVDD  | /         | 55.80    | 57.59         |

## Acknowledgments

- Paper: "Quantum support vector data description for anomaly detection"
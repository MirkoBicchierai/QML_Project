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

## Some Results

### QSVDD FMNIST

#### Latent Dimension 9, Noise 3D PCA
![QSVDD FMNIST Latent Dim 9 Noise 3D PCA](Result/QSVDD/fmnist/_lat-dim_9NOISE-3Dpca_0.pdf)

#### Latent Dimension 9, Noise 2D PCA
![QSVDD FMNIST Latent Dim 9 Noise 2D PCA](Result/QSVDD/fmnist/_lat-dim_9NOISE-2Dpca_0.pdf)

## License

This project is licensed under the [License Name] License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Paper: "Quantum support vector data description for anomaly detection"
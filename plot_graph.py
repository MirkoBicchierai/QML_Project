import numpy as np
from matplotlib import pyplot as plt

"""
Result obtained by the test execute on QSVDD Model (see at comet_ml link: https://www.comet.com/mirkobicchierai/qml/view/new/panels), 
copied in a py dict to plot the result
"""

mnist_results = {
    "1": {
        "w/o Noise": {
            "AUC": [78.95, 87.31, 53.22, 68.68, 69.82, 63.13, 70.62, 69.15, 73.23, 73.34],
            "Acc": [75.26, 80.52, 60.52, 67.89, 71.57, 64.73, 70.00, 67.89, 69.47, 70.00]
        },
        "w Noise": {
            "AUC": [70.66, 79.40, 65.68, 57.73, 62.98, 62.98, 69.76, 54.76, 72.74, 59.20],
            "Acc": [68.42, 75.26, 64.73, 57.36, 64.73, 56.84, 70.52, 57.89, 68.94, 63.15]
        }
    },
    "3": {
        "w/o Noise": {
            "AUC": [88.98, 93.16, 64.20, 87.47, 76.95, 52.57, 75.52, 81.26, 74.38, 66.74],
            "Acc": [83.15, 86.84, 64.73, 85.26, 72.63, 57.89, 73.68, 76.84, 70.52, 65.78]
        },
        "w Noise": {
            "AUC": [71.05, 67.86, 65.80, 62.77, 66.08, 64.48, 75.15, 61.92, 63.40, 75.57],
            "Acc": [72.31, 59.15, 68.47, 63.68, 58.94, 64.73, 76.00, 61.94, 64.31, 72.10]
        }
    },
    "9": {
        "w/o Noise": {
            "AUC": [92.33, 98.00, 77.28, 83.81, 69.71, 60.71, 67.90, 75.73, 80.26, 82.15],
            "Acc": [90.52, 95.26, 72.63, 78.42, 67.36, 60.52, 65.26, 71.57, 76.84, 77.36]
        },
        "w Noise": {
            "AUC": [83.55, 77.34, 72.23, 58.40, 68.25, 61.80, 84.33, 69.83, 64.56, 68.78],
            "Acc": [80.00, 73.68, 70.00, 57.89, 68.94, 62.10, 78.42, 66.31, 65.78, 68.78]
        }
    },
    "12": {
        "w/o Noise": {
            "AUC": [96.13, 90.45, 73.26, 74.78, 74.83, 63.56, 83.06, 72.98, 89.50, 74.98],
            "Acc": [90.52, 82.63, 66.84, 70.52, 68.94, 62.63, 78.42, 68.42, 82.63, 71.05]
        },
        "w Noise": {
            "AUC": [66.42, 58.32, 65.94, 58.40, 74.22, 60.68, 75.25, 58.94, 66.92, 69.85],
            "Acc": [64.21, 57.36, 64.21, 57.89, 73.15, 61.05, 73.68, 60.00, 66.31, 67.36]
        }
    }
}


fmnist_results = {
    "1": {
        "w/o Noise": {
            "AUC": [71.33, 95.67, 81.90, 88.08, 83.23, 49.62, 74.50, 91.73, 71.32, 91.75],
            "Acc": [70.00, 92.10, 78.42, 85.78, 83.15, 56.31, 75.78, 88.94, 69.47, 90.52]
        },
        "w Noise": {
            "AUC": [48.47, 80.75, 77.75, 86.45, 71.56, 72.04, 70.52, 81.03, 61.21, 88.16],
            "Acc": [55.78, 77.89, 76.31, 83.15, 71.05, 70.00, 67.36, 81.57, 61.57, 84.73]
        }
    },
    "3": {
        "w/o Noise": {
            "AUC": [86.16, 95.24, 82.20, 81.86, 80.34, 55.20, 68.92, 94.68, 74.25, 97.66],
            "Acc": [82.10, 90.00, 80.52, 77.36, 75.78, 56.31, 74.73, 91.05, 67.89, 92.63]
        },
        "w Noise": {
            "AUC": [76.15, 64.41, 81.11, 62.20, 75.27, 72.91, 78.27, 92.96, 68.32, 83.93],
            "Acc": [74.73, 65.78, 77.36, 62.10, 72.10, 71.57, 73.68, 89.47, 63.68, 80.52]
        }
    },
    "9": {
        "w/o Noise": {
            "AUC": [89.79, 95.21, 87.55, 88.74, 89.00, 76.77, 79.55, 96.14, 91.86, 90.81],
            "Acc": [84.21, 90.00, 84.73, 82.10, 83.68, 71.05, 77.89, 92.63, 86.31, 83.68]
        },
        "w Noise": {
            "AUC": [70.87, 87.01, 80.73, 75.81, 80.95, 65.00, 74.67, 78.95, 64.28, 71.64],
            "Acc": [69.47, 81.05, 79.47, 67.36, 78.42, 66.84, 74.73, 75.26, 62.63, 65.78]
        }
    },
    "12": {
        "w/o Noise": {
            "AUC": [84.25, 91.60, 82.03, 93.48, 87.28, 92.42, 83.65, 91.35, 93.92, 90.83],
            "Acc": [77.36, 87.36, 79.47, 86.84, 80.00, 88.42, 78.42, 91.63, 87.36, 88.97]
        },
        "w Noise": {
            "AUC": [67.00, 87.63, 58.45, 83.92, 74.23, 50.08, 67.32, 53.46, 77.64, 59.48],
            "Acc": [64.73, 82.10, 58.94, 78.42, 72.10, 55.26, 67.36, 57.36, 72.10, 57.89]
        }
    }
}

kmnist_results = {
    "1": {
        "w/o Noise": {
            "AUC": [61.28, 50.13, 56.91, 76.51, 59.58, 55.64, 76.31, 78.15, 54.07, 54.94],
            "Acc": [60.00, 56.84, 58.42, 71.57, 62.63, 55.78, 74.21, 63.15, 56.31, 75.78]
        },
        "w Noise": {
            "AUC": [52.71, 52.86, 51.34, 50.86, 50.15, 51.30, 65.38, 50.25, 57.32, 50.07],
            "Acc": [54.73, 53.68, 53.68, 54.21, 50.52, 62.10, 50.52, 60.00, 55.78, 51.03]
        }
    },
    "3": {
        "w/o Noise": {
            "AUC": [67.68, 69.27, 57.56, 55.36, 61.14, 60.17, 70.27, 76.68, 53.15, 52.28],
            "Acc": [64.21, 69.47, 61.05, 54.21, 60.00, 58.94, 71.05, 71.05, 58.42, 53.68]
        },
        "w Noise": {
            "AUC": [53.26, 54.46, 50.68, 53.07, 58.83, 60.31, 61.23, 64.44, 51.26, 56.27],
            "Acc": [56.84, 55.78, 55.78, 55.94, 58.42, 60.47, 62.63, 65.78, 55.78, 56.84]
        }
    },
    "9": {
        "w/o Noise": {
            "AUC": [75.22, 74.72, 67.69, 67.72, 66.45, 40.83, 76.74, 69.50, 57.17, 65.60],
            "Acc": [70.00, 69.47, 65.79, 64.73, 64.73, 53.15, 71.58, 66.84, 61.05, 62.10]
        },
        "w Noise": {
            "AUC": [57.91, 61.65, 67.55, 53.35, 58.90, 55.63, 58.86, 51.12, 56.70, 53.84],
            "Acc": [56.84, 61.57, 66.84, 56.31, 61.57, 58.42, 61.05, 54.73, 61.05, 57.89]
        }
    },
    "12": {
        "w/o Noise": {
            "AUC": [66.57, 49.32, 61.80, 62.50, 66.63, 63.00, 81.54, 77.51, 48.74, 72.98],
            "Acc": [63.15, 54.21, 62.10, 62.10, 61.05, 63.15, 76.84, 76.31, 55.78, 71.57]
        },
        "w Noise": {# classe 1, 6,7, 8
            "AUC": [55.48, 55.46, 57.53, 57.27, 51.70, 55.94, 61.35, 62.26, 61.56, 54.27],
            "Acc": [57.89, 59.52, 59.47, 56.31, 52.10, 54.21, 60.73, 62.89, 61.78, 56.31]
        }
    }
}

cifar10_results = {
    "1": {
        "w/o Noise": {
            "AUC": [67.35, 65.12, 50.65, 50.84, 59.28, 51.91, 51.62, 55.96, 54.56, 67.78],
            "Acc": [66.84, 65.78, 55.26, 51.10, 61.57, 54.21, 52.63, 59.47, 59.89, 66.31]
        },
        "w Noise": {
            "AUC": [63.67, 52.97, 50.25, 56.41, 53.92, 50.36, 56.75, 51.75, 53.56, 56.11],
            "Acc": [62.63, 56.84, 53.15, 61.05, 57.36, 55.26, 58.94, 52.63, 58.94, 56.84]
        }
    },
    "3": {
        "w/o Noise": {
            "AUC": [73.82, 59.27, 52.16, 54.75, 50.82, 50.01, 61.57, 50.40, 56.23, 67.73],
            "Acc": [70.00, 60.52, 54.73, 55.78, 50.96, 50.47, 63.15, 54.21, 59.47, 63.15]
        },
        "w Noise": {
            "AUC": [54.58, 54.72, 54.26, 55.63, 56.40, 55.54, 59.02, 52.61, 47.90, 67.77],
            "Acc": [58.42, 56.31, 57.89, 55.26, 58.42, 60.00, 58.94, 54.21, 53.68, 64.73]
        }
    },
    "9": {
        "w/o Noise": {
            "AUC": [70.58, 59.96, 62.57, 51.54, 59.24, 59.36, 58.56, 58.45, 63.24, 63.77],
            "Acc": [67.36, 56.78, 65.78, 54.73, 61.05, 60.00, 60.52, 59.42, 62.10, 62.63]
        },
        "w Noise": {
            "AUC": [64.04, 51.70, 61.46, 54.34, 54.26, 56.51, 59.51, 53.63, 69.56, 54.40],
            "Acc": [63.15, 55.78, 64.21, 55.78, 55.63, 58.94, 61.57, 55.26, 66.31, 59.47]
        }
    },
    "12": {
        "w/o Noise": {
            "AUC": [68.62, 59.62, 64.06, 47.33, 59.71, 60.18, 52.20, 50.97, 60.63, 67.26],
            "Acc": [66.84, 61.05, 65.26, 52.10, 61.05, 60.52, 56.31, 54.73, 62.63, 68.42]
        },
        "w Noise": {
            "AUC": [59.40, 44.04, 64.06, 44.45, 53.16, 58.76, 59.18, 50.64, 62.28, 62.04],
            "Acc": [61.05, 50.00, 65.26, 48.94, 53.68, 61.05, 60.00, 55.26, 58.94, 63.15]
        }
    }
}

"""
Reads the data, computes the mean AUC, and plots the curve.
Takes as input the dataset name, the dictionary to analyze and the color to use.
"""
def plot_auc_vs_latent_dimension(dataset, dataset_name, color):
    latent_dimensions = []
    mean_auc_wo_noise = []
    std_auc_wo_noise = []
    mean_auc_w_noise = []
    std_auc_w_noise = []

    for dim, values in dataset.items():
        latent_dimensions.append(int(dim))
        auc_wo_noise = values["w/o Noise"]["AUC"]
        auc_w_noise = values["w Noise"]["AUC"]

        mean_auc_wo_noise.append(np.mean(auc_wo_noise))
        std_auc_wo_noise.append(np.std(auc_wo_noise))

        mean_auc_w_noise.append(np.mean(auc_w_noise))
        std_auc_w_noise.append(np.std(auc_w_noise))

    sorted_indices = np.argsort(latent_dimensions)
    latent_dimensions = np.array(latent_dimensions)[sorted_indices]

    mean_auc_wo_noise = np.array(mean_auc_wo_noise)[sorted_indices]
    #std_auc_wo_noise = np.array(std_auc_wo_noise)[sorted_indices]

    mean_auc_w_noise = np.array(mean_auc_w_noise)[sorted_indices]
    #std_auc_w_noise = np.array(std_auc_w_noise)[sorted_indices]

    plt.plot(latent_dimensions, mean_auc_wo_noise, marker='o', linestyle='-', label=f'w/o Noise {dataset_name}',
             color=color)
    #plt.fill_between(latent_dimensions, mean_auc_wo_noise - std_auc_wo_noise, mean_auc_wo_noise + std_auc_wo_noise,color=color, alpha=0.2)

    plt.plot(latent_dimensions, mean_auc_w_noise, marker='s', linestyle='--', label=f'w Noise {dataset_name}',
             color=color)
    #plt.fill_between(latent_dimensions, mean_auc_w_noise - std_auc_w_noise, mean_auc_w_noise + std_auc_w_noise, color=color, alpha=0.2)
    plt.xticks(latent_dimensions, [str(dim) for dim in latent_dimensions])


""" Save the plot for each dataset in the Result Folder"""
def main():
    plt.figure(figsize=(8, 6))
    plot_auc_vs_latent_dimension(fmnist_results, "FMNIST", "#1f77b4")
    plot_auc_vs_latent_dimension(mnist_results, "MNIST", "#d62728")
    plot_auc_vs_latent_dimension(kmnist_results, "KMNIST", "#ff7f0e")
    plot_auc_vs_latent_dimension(cifar10_results, "CIFAR10", "#2ca02c")

    plt.xlabel("Latent Dimension")
    plt.ylabel("AUC Score")
    plt.title("AUC Score vs. Latent Dimension")
    plt.legend()
    plt.grid(True)
    plt.xlim(left=1)

    plt.savefig("Result/AUC Score vs. Latent Dimension.png")
    plt.savefig("Result/AUC Score vs. Latent Dimension.pdf")


if __name__ == "__main__":
    main()


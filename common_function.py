import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

"""
Function to select the dataset based on the given name.
Takes the dataset name as input and returns two splits: one for training and one for testing.

Available options: cifar10, mnist, fmnist, kmnist.
All images are resized to 16x16, and for cifar10, a Grayscale transform is applied.
"""

def select_dataset(dataset):
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor()
        ])

    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root='Dataset/', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='Dataset/', train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    elif dataset == 'fmnist':
        train_dataset = datasets.FashionMNIST(root='Dataset/', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='Dataset/', train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    elif dataset == 'kmnist':
        train_dataset = datasets.KMNIST(root='Dataset/', train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root='Dataset/', train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='Dataset/', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='Dataset/', train=False, download=True, transform=transform)
        return train_dataset, test_dataset


"""
Function to correctly split the dataset obtained from the select_dataset function.

- For the training dataset: selects only train_samples from the specified target_class.
- For the test dataset: selects test_samples from the target_class and test_samples_other from the other classes.

Returns the final dataloaders: train_loader and test_loader.
"""

def data(dataset, target_class, batch_size, train_samples, test_samples_target, test_samples_other):
    train_dataset, test_dataset = select_dataset(dataset)

    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)

    # Get indices for target class samples
    train_target_indices = np.where(train_targets == target_class)[0]
    test_target_indices = np.where(test_targets == target_class)[0]

    # Randomly sample required number of target class indices
    train_selected_indices = np.random.choice(
        train_target_indices,
        size=min(train_samples, len(train_target_indices)),
        replace=False
    )

    # For test set, gather both target and other class samples
    test_selected_indices = []

    # Add target class samples
    test_selected_indices.extend(
        np.random.choice(
            test_target_indices,
            size=min(test_samples_target, len(test_target_indices)),
            replace=False
        )
    )

    # Add samples from other classes
    other_classes = [i for i in range(10) if i != target_class]
    for other_class in other_classes:
        other_indices = np.where(test_targets == other_class)[0]
        test_selected_indices.extend(
            np.random.choice(
                other_indices,
                size=min(test_samples_other, len(other_indices)),
                replace=False
            )
        )

    # Create subset datasets
    train_subset = Subset(train_dataset, train_selected_indices)
    test_subset = Subset(test_dataset, test_selected_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=12
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=12
    )

    return train_loader, test_loader

"""
This function plot the ROC curve 
"""

def auc_plot(auc, fpr, tpr, path):
    print(f"AUC: {auc:.4f}")

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='dodgerblue', lw=lw, label="{:.2f}".format(auc * 100))
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()

"""
This function plot a tensor as images
"""
def plot_tensor(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()

    batch_size = tensor.shape[0]
    num_samples = batch_size

    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    if num_samples > 1:
        axes = axes.flatten()

    for i in range(num_samples):
        if num_samples > 1:
            ax = axes[i]
        else:
            ax = axes
        im = ax.imshow(tensor[i], cmap='gray')
        ax.axis('off')

    for i in range(num_samples, len(axes) if num_samples > 1 else 1):
        if num_samples > 1:
            fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

"""
This function executes the PCA algorithm on the outputs from the model, with k = 2 and k = 3.
It then calls pca2_plot and pca3_plot to plot the hypersphere of the PCA in 2D and 3D space.
"""
def plot_quantum_sphere(train_measures, test_measures, test_labels, target_class, path):
    train_np = train_measures.cpu().numpy()
    test_measures = torch.cat(test_measures)
    test_labels = torch.cat(test_labels)
    test_np = test_measures.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    target_mask = test_labels_np == target_class

    combined_data = np.vstack([train_np, test_np])

    pca = PCA(n_components=2)
    combined_projected = pca.fit_transform(combined_data)
    train_projected = combined_projected[:len(train_np)]
    test_projected = combined_projected[len(train_np):]
    pca2_plot(train_projected, test_projected, target_mask, pca, target_class, path+"2Dpca_"+str(target_class)+".pdf")

    pca = PCA(n_components=3)
    combined_projected = pca.fit_transform(combined_data)
    train_projected = combined_projected[:len(train_np)]
    test_projected = combined_projected[len(train_np):]
    pca3_plot(train_projected, test_projected, target_mask, pca, target_class, path+"3Dpca_"+str(target_class)+".pdf")

"""
This function plot the hypersphere of the 2D PCA and save it in the 'path' folder.
"""
def pca2_plot(train_projected, test_projected, target_mask, pca, target_class, path):
    plt.figure(figsize=(10, 8))

    plt.scatter(train_projected[:, 0],
                train_projected[:, 1],
                c='blue',
                alpha=0.6,
                label='Training Set')

    plt.scatter(test_projected[target_mask, 0],
                test_projected[target_mask, 1],
                c='green',
                alpha=0.6,
                label=f'Test Set (Class {target_class})')

    plt.scatter(test_projected[~target_mask, 0],
                test_projected[~target_mask, 1],
                c='red',
                alpha=0.6,
                label='Test Set (Other Classes)')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

    plt.title('Quantum Measurements Projected to 2D')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

"""
This function plot the hypersphere of the 3D PCA and save it in the 'path' folder.
"""
def pca3_plot(train_projected, test_projected, target_mask, pca, target_class, path):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(train_projected[:, 0],
               train_projected[:, 1],
               train_projected[:, 2],
               c='blue',
               alpha=0.6,
               label='Training Set')

    ax.scatter(test_projected[target_mask, 0],
               test_projected[target_mask, 1],
               test_projected[target_mask, 2],
               c='green',
               alpha=0.6,
               label=f'Test Set (Class {target_class})')

    ax.scatter(test_projected[~target_mask, 0],
               test_projected[~target_mask, 1],
               test_projected[~target_mask, 2],
               c='red',
               alpha=0.6,
               label='Test Set (Other Classes)')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')

    plt.title(f'Quantum Measurements 3D (Target Class: {target_class})')
    plt.legend()
    ax.grid(True)
    plt.savefig(path)
    plt.close()

"""
This function plot parameters convergence and save it in the 'path' folder.
"""
def plot_parameters(param_history, path, d, c):
    for i in range(param_history.shape[1]):
        plt.plot(param_history[:, i], label=f'Param {i}')
    plt.xlabel("Epochs")
    plt.ylabel("Parameters")
    plt.title("Parameters convergence - " + d + " - " + str(c))
    plt.savefig(path)
    plt.close()
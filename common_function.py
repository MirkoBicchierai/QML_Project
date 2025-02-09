import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def select_dataset(dataset):
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


def data(dataset, target_class, batch_size, train_samples, test_samples_target, test_samples_other):

    train_dataset, test_dataset = select_dataset(dataset)

    # Convert targets to numpy arrays for easier indexing
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

    # Create dataloaders
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

def auc_plot(auc, fpr, tpr):
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
    plt.show()

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
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from ModelQAE import QuantumAutoencoder
from common_function import auc_plot, plot_tensor, data, plot_quantum_sphere


def qae_loss_repo(trash_q_measurements):
    loss = 0
    for p in trash_q_measurements:
        loss = loss + (-torch.sum(p))
    loss = loss / len(trash_q_measurements)
    return loss

def loss_paper(trash_q_measurements):
    """
    L(θ) = n ∑(j=1 to t) (1 - measurements[j])
    Each measurement is already Trace(σz ρ) from qml.expval(qml.PauliZ(i)) in the model forward
    """
    batch_loss = torch.sum(1 - trash_q_measurements, dim=1)  # Sum over measurements
    return torch.mean(batch_loss)


def train(epochs, train_loader, model, optimizer, device, input_size):
    model.train()
    loss_history = []
    param_history = []
    current_params = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    param_history.append(np.concatenate(current_params))

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)

            optimizer.zero_grad()
            trash_q_measurements = model(inputs.view(-1, input_size))
            loss = loss_paper(trash_q_measurements)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        current_params = [p.data.cpu().numpy().flatten() for p in model.parameters()]
        param_history.append(np.concatenate(current_params))
        print(f'Epoch {epoch} - Train loss: {avg_loss:.4f}')

    return loss_history, np.array(param_history)


def test(target_class, latent_dim, test_dataloader, train_dataloader, model):
    model.eval()

    with torch.no_grad():
        print(f"Starting test function with target_class={target_class}, latent_dim={latent_dim}, elaborating c")
        y_pred = []
        y_true = []

        train_set_measures = torch.cat([model(inputs) for inputs, _ in train_dataloader])
        c = torch.mean(train_set_measures, dim=0)

        print(f"Testing on test dataset")

        test_predictions = []
        test_labels = []

        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            pred = model(inputs)
            test_predictions.append(pred)
            test_labels.append(labels)
            plot_tensor(inputs.squeeze())
            for j in range(len(pred)):
                if labels[j] == target_class:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(0)
                else:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(1)

        plot_quantum_sphere(train_set_measures, test_predictions, test_labels, target_class)

        fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_pred))
        auc = roc_auc_score(np.array(y_true), np.array(y_pred))

        print("Test completed")

    return auc, y_pred, y_true, fpr, tpr, thresholds


def main():

    # Constants
    n_qubits = 8
    n_latent_qubits = 2
    n_trash_qubits = n_qubits - n_latent_qubits
    device = "cpu"

    # dataset parameters
    input_size = 16 * 16
    batch_size = 16
    target_class, train_samples, test_samples_target, test_samples_other = 0, 600, 100, 10
    dataset_name = 'mnist' # kmnist, fmnist

    # model parameters
    latent_dim = 9  # pauli observable
    n_layers = 10

    # train loop parameters
    epochs = 15
    lr = 0.001

    model = QuantumAutoencoder(n_layers, n_qubits, n_trash_qubits).to(device)
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader, test_dataloader = data(dataset_name, target_class, batch_size, train_samples, test_samples_target,test_samples_other)

    loss_history, param_history = train(epochs, train_dataloader, model, optimizer, device, input_size)

    torch.save(model, 'Models/QAE.pth')

    plt.style.use('default')
    plt.plot(loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss convergence")
    plt.show()

    for i in range(param_history.shape[1]):
        plt.plot(param_history[:, i], label=f'Param {i}')
    plt.xlabel("Iterations")
    plt.ylabel("Parameters")
    plt.title("Parameters convergence")
    plt.show()

    auc, y_pred, y_true, fpr, tpr, thresholds = test(target_class, latent_dim, test_dataloader, train_dataloader, model)

    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    binary_pred = [1 if d >= optimal_threshold else 0 for d in y_pred]
    accuracy = accuracy_score(y_true, binary_pred)

    print(f"Best accuracy: {accuracy:.2%}")

    auc_plot(auc, fpr, tpr)


if __name__ == "__main__":
    main()
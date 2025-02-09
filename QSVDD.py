import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from tqdm import tqdm
import numpy as np
from ModelQSVDD import QSVDDModel
from common_function import auc_plot, data, plot_tensor


def qsvdd_loss(y, predictions):
    loss = 0
    for l, p in zip(y, predictions):
        loss = loss + torch.sum((p - l) ** 2)
    loss = loss / len(y)
    return loss

def train(epochs, train_dataloader, model, optimizer, device, input_size):
    model.train()
    loss_history = []
    param_history = []
    current_params = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    param_history.append(np.concatenate(current_params))

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            measurements = model(inputs.view(-1, input_size))

            loss = qsvdd_loss(labels, measurements)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
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
        c = torch.mean(torch.cat([model(inputs) for inputs, _ in train_dataloader]), dim=0)
        print(f"Testing on test dataset")

        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            pred = model(inputs)
            plot_tensor(inputs.squeeze())
            for j in range(len(pred)):
                if labels[j] == target_class:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(0)
                else:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(1)

        fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_pred))
        auc = roc_auc_score(np.array(y_true), np.array(y_pred))

        print("Test completed")

    return auc, y_pred, y_true, fpr, tpr, thresholds

def main():

    # Constants
    n_qubits = 8
    device = "cpu"

    # dataset parameters
    input_size = 16 * 16
    batch_size = 16
    target_class, train_samples, test_samples_target, test_samples_other = 0, 600, 100, 10
    dataset_name = 'mnist'  # kmnist, fmnist

    # model parameters
    latent_dim = 9  # pauli observable
    n_layers = 5
    num_params_conv = 15

    # train loop parameters
    epochs = 100
    lr = 0.001

    model = QSVDDModel(n_layers, n_qubits, num_params_conv).to(device)
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader, test_dataloader = data(dataset_name, target_class, batch_size, train_samples, test_samples_target,
                                             test_samples_other)

    loss_history, param_history = train(epochs, train_dataloader, model, optimizer, device, input_size)

    torch.save(model, 'Models/QSVDD.pth')

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
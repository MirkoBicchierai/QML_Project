import comet_ml
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from ModelQAE import QuantumAutoencoder
from common_function import auc_plot, plot_tensor, data, plot_quantum_sphere, plot_parameters
from noise import noise_model


"""
Loss function for the QAE model implementing the loss described in the referenced paper.
L(θ) = n ∑(j=1 to t) (1 - measurements[j])
Each measurement is already Trace(σz ρ) from qml.expval(qml.PauliZ(i)) in the model forward
"""

def loss_paper(trash_q_measurements):

    batch_loss = torch.sum(1 - trash_q_measurements, dim=1)  # Sum over measurements
    return torch.mean(batch_loss)

"""
Train function that train the QAE Model for 'epochs' epochs.
return the loss history and the param history for the plot in main function. 
"""

def train(epochs, train_loader, model, optimizer, device, input_size, exp, dataset):
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
        exp.log_metric('QAE '+dataset+' - Train Loss', avg_loss, step=epoch)

    return loss_history, np.array(param_history)

"""
Test function that calculates the AUC and accuracy on the test set of the selected dataset in two modes: with noise and without noise.
Save the result in the base_path folder (PCA 2D/3D Plot)

Returns the AUC score, the predicted labels along with the true labels, and the thresholds of the ROC curve.
"""

def test(target_class, latent_dim, test_dataloader, train_dataloader, model, base_path):
    model.eval()

    with torch.no_grad():
        print(f"Starting test function with target_class={target_class}, latent_dim={latent_dim}, elaborating c")
        y_pred = []
        y_true = []

        train_set_measures = torch.cat([model(inputs) for inputs, _ in tqdm(train_dataloader)])
        c = torch.mean(train_set_measures, dim=0)

        print(f"Testing on test dataset")

        test_predictions = []
        test_labels = []

        for batch_idx, (inputs, labels) in enumerate(test_dataloader):
            pred = model(inputs)
            test_predictions.append(pred)
            test_labels.append(labels)
            #plot_tensor(inputs.squeeze())
            for j in range(len(pred)):
                if labels[j] == target_class:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(0)
                else:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(1)

        plot_quantum_sphere(train_set_measures, test_predictions, test_labels, target_class, base_path)

        fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_pred))
        auc = roc_auc_score(np.array(y_true), np.array(y_pred))

        print("Test completed")

    return auc, y_pred, y_true, fpr, tpr, thresholds

"""
Main function that executes a single experiment with the QAE model.
Takes as input the 'target' class (ranging from 0 to 9), the dataset name, and the path to save the results.
"""

def main(target, dataset, lat_dim, base_path, exp):
    # Constants
    n_qubits = 8
    n_latent_qubits = 2
    n_trash_qubits = n_qubits - n_latent_qubits
    device = "cpu"

    # dataset parameters
    input_size = 16 * 16
    batch_size = 4
    target_class, train_samples, test_samples_target, test_samples_other = target, 600, 100, 10
    dataset_name = dataset

    # model parameters
    latent_dim = lat_dim  # pauli observable
    n_layers = 10

    # train loop parameters
    epochs = 50
    lr = 0.001

    model = QuantumAutoencoder(n_layers, n_qubits, n_trash_qubits, noise=None).to(device)
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader, test_dataloader = data(dataset_name, target_class, batch_size, train_samples, test_samples_target,
                                             test_samples_other)

    loss_history, param_history = train(epochs, train_dataloader, model, optimizer, device, input_size, exp, dataset)

    torch.save(model, 'Models/QAE_' + str(target) + '_' + dataset + '.pth')

    plot_parameters(param_history, base_path + "parameters_" + str(target) + ".pdf", dataset, target)

    auc, y_pred, y_true, fpr, tpr, thresholds = test(target_class, latent_dim, test_dataloader, train_dataloader, model,
                                                     base_path)
    exp.log_metric(dataset + 'QAE Test AUC without Noise', auc)

    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    binary_pred = [1 if d >= optimal_threshold else 0 for d in y_pred]
    accuracy = accuracy_score(y_true, binary_pred)

    print(f"Best accuracy: {accuracy:.2%}")
    exp.log_metric(dataset + 'QAE Test Accuracy without Noise', accuracy)

    auc_plot(auc, fpr, tpr, base_path + "AUC_" + str(target) + ".pdf")

    model.noise = noise_model()

    auc, y_pred, y_true, fpr, tpr, thresholds = test(target_class, latent_dim, test_dataloader, train_dataloader, model, base_path+"NOISE-")
    exp.log_metric(dataset + 'QAE Test AUC with Noise', auc)

    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    binary_pred = [1 if d >= optimal_threshold else 0 for d in y_pred]
    accuracy = accuracy_score(y_true, binary_pred)

    print(f"Best accuracy: {accuracy:.2%}")
    exp.log_metric(dataset + 'QAE Test Accuracy with Noise', accuracy)

    auc_plot(auc, fpr, tpr, base_path + "AUC_NOISE_" + str(target) + ".pdf")

"""
Execute all tests with the QSVDD model for each dataset in ["mnist", "fmnist", "kmnist", "cifar10"].
"""

if __name__ == "__main__":
    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    datasets = ["kmnist", "fmnist", "mnist", "cifar10"]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for d in tqdm(datasets):
        for c in classes:
            print("Experiment with " + d +" class "+ str(c))
            experiment_name = f"QAE - {d}_class_{c}"
            exp = comet_ml.Experiment(project_name="qml", auto_metric_logging=False, auto_param_logging=False)
            exp.set_name(experiment_name)
            parameters = {'dataset': d, 'class': c}
            exp.log_parameters(parameters)
            main(target=c, dataset=d, lat_dim=9, base_path="Result/QAE/" + d + "/", exp=exp)

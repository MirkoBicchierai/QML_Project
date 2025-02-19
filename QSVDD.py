import comet_ml
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from tqdm import tqdm
import numpy as np
from ModelQSVDD import QSVDDModel
from common_function import auc_plot, data, plot_tensor, plot_quantum_sphere, plot_parameters
from noise import noise_model


"""
Train function that train the QSVDD Model for 'epochs' epochs.
return the loss history and the param history for the plot in main function. 
"""

def train(epochs, train_dataloader, model, optimizer, device, input_size, exp, dataset):
    model.train()
    loss_history = []
    param_history = []
    current_params = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    param_history.append(np.concatenate(current_params))

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
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
        exp.log_metric('QSVDD ' + dataset + ' - Train Loss', avg_loss, step=epoch)

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
            # plot_tensor(inputs.squeeze())
            for j in range(len(pred)):
                if labels[j] == target_class:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(0)
                else:
                    y_pred.append(((pred[j] - c) ** 2).mean().item())
                    y_true.append(1)

        if latent_dim > 1:
            plot_quantum_sphere(train_set_measures, test_predictions, test_labels, target_class, base_path)

        fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_pred))
        auc = roc_auc_score(np.array(y_true), np.array(y_pred))

        print("Test completed")

    return auc, y_pred, y_true, fpr, tpr, thresholds

"""
Loss function for the QSVDD model implementing the loss described in the referenced paper.
"""

def qsvdd_loss(center, predictions):
    loss = 0
    for l, p in zip(center, predictions):
        loss = loss + torch.sum((p - l) ** 2)
    loss = loss / len(center)
    return loss

"""
Main function that executes a single experiment with the QSVDD model.
Takes as input the 'target' class (ranging from 0 to 9), the latent dimension, the dataset name, and the path to save the results.
"""

def main(target, dataset, lat_dim, base_path, exp):
    # Constants
    n_qubits = 8
    device = "cpu"

    # dataset parameters
    input_size = 16 * 16
    batch_size = 16
    target_class, train_samples, test_samples_target, test_samples_other = target, 600, 100, 10
    dataset_name = dataset

    # model parameters
    latent_dim = lat_dim  # pauli observable
    n_layers = 5
    num_params_conv = 15

    # train loop parameters
    epochs = 20
    lr = 0.001

    model = QSVDDModel(n_layers, n_qubits, num_params_conv, noise=None, latent_dim=latent_dim).to(device)
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader, test_dataloader = data(dataset_name, target_class, batch_size, train_samples, test_samples_target,
                                             test_samples_other)

    loss_history, param_history = train(epochs, train_dataloader, model, optimizer, device, input_size, exp, dataset)

    torch.save(model, 'Models/QSVDD_' + str(target) + '_' + str(latent_dim) + '_' + dataset + '.pth')

    plot_parameters(param_history, base_path + "parameters_" + str(target) + "_lat-dim_" + str(latent_dim) + ".pdf",
                    dataset, target)

    auc, y_pred, y_true, fpr, tpr, thresholds = test(target_class, latent_dim, test_dataloader, train_dataloader, model,
                                                     base_path + "_lat-dim_" + str(latent_dim))
    exp.log_metric(dataset + 'QSVDD Test AUC without Noise - latent dimension ' + str(latent_dim), auc)

    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    binary_pred = [1 if d >= optimal_threshold else 0 for d in y_pred]
    accuracy = accuracy_score(y_true, binary_pred)

    print(f"Best accuracy: {accuracy:.2%}")
    exp.log_metric(dataset + 'QSVDD Test Accuracy without Noise - latent dimension ' + str(latent_dim), accuracy)

    auc_plot(auc, fpr, tpr, base_path + "AUC_" + "lat-dim_" + str(latent_dim) + "_" + str(target) + ".pdf")

    model.noise = noise_model()

    auc, y_pred, y_true, fpr, tpr, thresholds = test(target_class, latent_dim, test_dataloader, train_dataloader, model,
                                                     base_path + "_lat-dim_" + str(latent_dim) + "NOISE-")
    exp.log_metric(dataset + 'QSVDD Test AUC with Noise - latent dimension ' + str(latent_dim), auc)

    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    binary_pred = [1 if d >= optimal_threshold else 0 for d in y_pred]
    accuracy = accuracy_score(y_true, binary_pred)

    print(f"Best accuracy: {accuracy:.2%}")
    exp.log_metric(dataset + 'QSVDD Test Accuracy with Noise - latent dimension ' + str(latent_dim), accuracy)

    auc_plot(auc, fpr, tpr, base_path + "AUC_NOISE_" + "lat-dim_" + str(latent_dim) + "_" + str(target) + ".pdf")


"""
Execute all tests with the QSVDD model for each dataset in ["mnist", "fmnist", "kmnist", "cifar10"] and each latent dimension in [1, 3, 9, 12].
"""

if __name__ == "__main__":
    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    datasets = ["mnist", "fmnist", "kmnist", "cifar10"]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    latent_dimensions = [1, 3, 9, 12]
    for lat in latent_dimensions:
        for d in tqdm(datasets):
            for c in classes:
                print("Experiment with " + d + " class " + str(c) + " latent dimension " + str(lat))
                experiment_name = f"QSVDD - {d}_class_{c}_latent_dimension_{lat}"
                exp = comet_ml.Experiment(project_name="qml", auto_metric_logging=False, auto_param_logging=False)
                exp.set_name(experiment_name)
                parameters = {'dataset': d, 'class': c, 'latent': lat}
                exp.log_parameters(parameters)
                main(target=c, dataset=d, lat_dim=lat, base_path="Result/QSVDD/" + d + "/", exp=exp)

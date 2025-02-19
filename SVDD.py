import comet_ml
import numpy as np
import torch
from tqdm import tqdm
from ModelSVDD import SVDD
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from common_function import auc_plot, data

"""
Function to initialize the center c before training.
Returns the center.
"""

def initialize_center_c(dataloader, model):
    model.eval()
    with torch.no_grad():
        center = torch.zeros(model.latent_dim, device=next(model.parameters()).device)
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.cuda()
            outputs = model(inputs)
            center += torch.sum(outputs, dim=0)
            center /= outputs.shape[0]
            break
    model.train()
    return center

"""
Train function that train the QSVDD Model for 'epochs' epochs.
The center c, the loss parameter, is initialized with initialize_center_c
Return the center c and the param history for the plot in main function. 
"""

def train(model, optimizer, train_dataloader, epochs, dataset, exp):
    loss_history = []
    c = initialize_center_c(train_dataloader, model)
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch_idx, (inputs, _) in enumerate(train_dataloader):
            inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.mean(torch.sum((outputs - c) ** 2, dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        loss_history.append(avg_loss)

        print(f'Epoch {epoch} - Train loss: {avg_loss:.4f}')
        exp.log_metric('SVDD ' + dataset + ' - Train Loss', avg_loss, step=epoch)

    return c, loss_history

"""
Test function that calculates the AUC and accuracy on the test set of the selected dataset.
Return the AUC score, Accuracy score and fpr, tpr
"""

def test(model, test_dataloader, target_class, c):
    y_true = []
    y_pred = []
    distances = []
    model = model.cuda()

    with torch.no_grad():
        for data, labels in test_dataloader:
            inputs = data.cuda()
            labels = labels.cuda()

            outputs = model(inputs.float())
            dist = torch.sum((outputs - c) ** 2, dim=1)

            for label, d in zip(labels.cpu().numpy(), dist.cpu().numpy()):
                if label == target_class:
                    y_true.append(0)
                else:
                    y_true.append(1)
                distances.append(d.item())
                y_pred.append(d.item())

    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    binary_pred = [1 if d >= optimal_threshold else 0 for d in distances]
    accuracy = accuracy_score(y_true, binary_pred)

    return auc, fpr, tpr, accuracy

"""
Main function that executes a single experiment with the SVDD model.
Takes as input the 'target' class (ranging from 0 to 9), the dataset name, and the path to save the results.
"""

def main(target, dataset, lat_dim, base_path, exp):
    dataset = dataset
    latent_dim = lat_dim
    epochs = 50
    learning_rate = 0.001
    batch_size = 16
    weight_decay = 0.001

    target_class, train_samples, test_samples_target, test_samples_other = target, 600, 100, 10

    train_dataloader, test_dataloader = data(dataset, target_class, batch_size, train_samples, test_samples_target,
                                             test_samples_other)

    model = SVDD(latent_dim=latent_dim).cuda()
    # summary(model, (1, 16, 16))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    c, loss_history = train(model, optimizer, train_dataloader, epochs, dataset, exp)

    auc, fpr, tpr, accuracy = test(model, test_dataloader, target_class, c)
    print(f"Accuracy: {accuracy:.4f}")

    exp.log_metric(dataset + 'SVDD Test AUC', auc)
    exp.log_metric(dataset + 'SVDD Test Accuracy', accuracy)

    auc_plot(auc, fpr, tpr, base_path + "AUC_" + str(target) + ".pdf")

"""
Execute all tests with the SVDD model for each dataset in ["mnist", "fmnist", "kmnist", "cifar10"].
"""

if __name__ == '__main__':
    comet_ml.login(api_key="S8bPmX5TXBAi6879L55Qp3eWW")
    datasets = ["kmnist", "fmnist", "mnist", "cifar10"]
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for d in tqdm(datasets):
        for c in classes:
            print("Experiment with " + d + " class " + str(c))
            experiment_name = f"SVDD - {d}_class_{c}"
            exp = comet_ml.Experiment(project_name="qml", auto_metric_logging=False, auto_param_logging=False)
            exp.set_name(experiment_name)
            parameters = {'dataset': d, 'class': c}
            exp.log_parameters(parameters)
            main(target=c, dataset=d, lat_dim=9, base_path="Result/SVDD/" + d + "/", exp=exp)

import numpy as np
import torch
from torchsummary import summary
from ModelSVDD import SVDD
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from common_function import auc_plot, data


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

def train(model, optimizer,train_dataloader, epochs, c):
    loss_history = []
    for epoch in range(epochs):
        data = next(iter(train_dataloader))
        inputs, _ = data
        inputs = inputs.cuda()
        optimizer.zero_grad()

        if c is None:
            c = initialize_center_c(train_dataloader, model)

        outputs = model(inputs)
        loss = torch.mean(torch.sum((outputs - c) ** 2, dim=1))

        """ 
        l2_reg = torch.tensor(0., device=inputs.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg
        """

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print('Step: {} | Loss: {:.6f}'.format(epoch, loss.data))
    return c, loss_history

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

def main():
    dataset = 'mnist'
    latent_dim = 9
    epochs = 500
    learning_rate = 0.001
    batch_size = 16
    weight_decay = 0.001
    c = None
    target_class, train_samples, test_samples_target, test_samples_other = 0, 600, 100, 10

    train_dataloader, test_dataloader  = data(dataset,target_class, batch_size, train_samples, test_samples_target, test_samples_other)

    model = SVDD(latent_dim=latent_dim).cuda()
    summary(model, (1, 16, 16))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    c, loss_history = train(model, optimizer, train_dataloader, epochs, c)

    plt.style.use('default')
    plt.plot(loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss convergence")
    plt.show()

    auc, fpr, tpr, accuracy = test(model, test_dataloader, target_class, c)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

    auc_plot(auc, fpr, tpr)

if __name__ == '__main__':
    main()

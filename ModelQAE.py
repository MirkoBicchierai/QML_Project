from itertools import combinations
import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 8
dev = qml.device("lightning.qubit", wires=n_qubits) # "default.qubit"

class QuantumAutoencoder(nn.Module):
    def __init__(self, n_layers, n_qubits, n_trash_qubits):
        super().__init__()
        self.quantum_weights = nn.Parameter(torch.randn((n_layers*n_qubits) - (n_qubits - n_trash_qubits), dtype=torch.float32))
        print("Weights", self.quantum_weights.shape)
        self.n_qubits, self.n_layers, self.n_trash_qubits = n_qubits, n_layers, n_trash_qubits

    def forward(self, x):
        b_s = x.shape[0]
        trash_measurements = []

        for i in range(b_s):
            measurements = quantum_circuit(x[i].view(-1), self.quantum_weights, self.n_qubits, self.n_layers, self.n_trash_qubits)
            trash_measurements.append(torch.stack(measurements))

        return torch.stack(trash_measurements)


@qml.qnode(dev, interface="torch") # , diff_method = 'backprop'
def quantum_circuit(inputs, weights, n_qubits, n_layers, n_trash_qubits):

    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)

    end_idx = 0
    for l in range(n_layers - 1):
        start_idx = l * n_qubits
        end_idx = (l + 1) * n_qubits
        layer(weights[start_idx:end_idx], n_qubits, n_trash_qubits)

    last_layer(weights[end_idx:len(weights)], n_trash_qubits)

    # Measurements
    result = [qml.expval(qml.PauliX(6)), qml.expval(qml.PauliY(6)), qml.expval(qml.PauliZ(6)),
              qml.expval(qml.PauliX(7)), qml.expval(qml.PauliY(7)), qml.expval(qml.PauliZ(7)),
              qml.expval(qml.PauliX(6) @ qml.PauliX(7)), qml.expval(qml.PauliY(6) @ qml.PauliY(7)),
              qml.expval(qml.PauliZ(6) @ qml.PauliZ(7))]

    return result

def layer(params, n_qubits, n_trash_qubits):  # params: 14
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)

    for i, j in combinations(range(0, n_trash_qubits), 2):
        qml.CZ(wires=[i, j])

    for idx in range(n_trash_qubits):
        for i in range(n_trash_qubits):
            for j in range(n_trash_qubits + i, n_qubits, n_trash_qubits):
                qml.CZ(wires=[(idx + i) % (n_trash_qubits), j])


def last_layer(params, n_trash_qubits):
    for i in range(n_trash_qubits):
        qml.RY(params[i], wires=i)


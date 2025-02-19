from itertools import combinations
import torch
import torch.nn as nn
import pennylane as qml
from matplotlib import pyplot as plt

""" QAE Model that follow the paper implementations. """
class QuantumAutoencoder(nn.Module):
    def __init__(self, n_layers, n_qubits, n_trash_qubits, noise):
        super().__init__()
        self.quantum_weights = nn.Parameter(torch.randn((n_layers*n_qubits) - (n_qubits - n_trash_qubits), dtype=torch.float32))# pi -pi
        print("Weights", self.quantum_weights.shape)
        self.n_qubits, self.n_layers, self.n_trash_qubits = n_qubits, n_layers, n_trash_qubits
        self.noise = noise

    def forward(self, x):
        b_s = x.shape[0]
        trash_measurements = []

        if self.noise:
            dev = qml.device("default.mixed", wires=self.n_qubits)
        else:
            dev = qml.device("lightning.qubit", wires=self.n_qubits)

        node = qml.QNode(quantum_circuit, dev, interface="torch")
        circuit = qml.add_noise(node, noise_model=self.noise) if self.noise else node

        for i in range(b_s):
            measurements = circuit(x[i].view(-1), self.quantum_weights, self.n_qubits, self.n_layers, self.n_trash_qubits)
            trash_measurements.append(torch.stack(measurements))

        return torch.stack(trash_measurements)

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

"""Plot the circuit to test"""
if __name__ == '__main__':

    print(qml.draw_mpl(quantum_circuit)(torch.ones(1, 256), torch.zeros(78), 8, 10, 6))
    plt.show()


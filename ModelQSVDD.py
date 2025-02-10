import torch
import torch.nn as nn
import pennylane as qml

class QSVDDModel(nn.Module):
    def __init__(self, n_layers, n_qubits, num_params_conv, noise, latent_dim):
        super().__init__()
        self.quantum_weights = nn.Parameter(torch.randn((n_layers*num_params_conv), dtype=torch.float32))
        print("Weights", self.quantum_weights.shape)
        self.n_qubits = n_qubits
        self.num_params_conv = num_params_conv
        self.noise = noise
        self.latent_dim = latent_dim

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
            measurements = circuit(x[i].view(-1), self.quantum_weights, self.n_qubits, self.num_params_conv, self.latent_dim)
            trash_measurements.append(torch.stack(measurements))

        return torch.stack(trash_measurements)



def quantum_circuit(inputs, weights, n_qubits, num_params_conv, latent_dim):

    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)

    conv_layer_1(U_SU4, weights[0:num_params_conv])
    conv_layer_1(U_SU4, weights[num_params_conv: 2 * num_params_conv])
    conv_layer_2(U_SU4, weights[2 * num_params_conv: 3 * num_params_conv])
    conv_layer_2(U_SU4, weights[3 * num_params_conv: 4 * num_params_conv])
    conv_layer_3(U_SU4, weights[4 * num_params_conv: 5 * num_params_conv])

    result = []
    if latent_dim == 1:
        result = [qml.expval(qml.PauliZ(6))]
    elif latent_dim == 3:
        result = [qml.expval(qml.PauliX(2)), qml.expval(qml.PauliY(2)), qml.expval(qml.PauliZ(2))]
    elif latent_dim == 6:
        result = [qml.expval(qml.PauliX(2)), qml.expval(qml.PauliY(2)), qml.expval(qml.PauliZ(2)),
                  qml.expval(qml.PauliX(6)), qml.expval(qml.PauliY(6)), qml.expval(qml.PauliZ(6))]
    elif latent_dim == 9:
        result = [qml.expval(qml.PauliX(2)), qml.expval(qml.PauliY(2)), qml.expval(qml.PauliZ(2)),
                  qml.expval(qml.PauliX(6)), qml.expval(qml.PauliY(6)), qml.expval(qml.PauliZ(6)),
                  qml.expval(qml.PauliX(2) @ qml.PauliX(6)), qml.expval(qml.PauliY(2) @ qml.PauliY(6)),
                  qml.expval(qml.PauliZ(2) @ qml.PauliZ(6))]
    elif latent_dim == 12:
        result = [qml.expval(qml.PauliX(2)), qml.expval(qml.PauliY(2)), qml.expval(qml.PauliZ(2)),
                  qml.expval(qml.PauliX(6)), qml.expval(qml.PauliY(6)), qml.expval(qml.PauliZ(6)),
                  qml.expval(qml.PauliX(2) @ qml.PauliX(6)), qml.expval(qml.PauliY(2) @ qml.PauliY(6)),
                  qml.expval(qml.PauliZ(2) @ qml.PauliZ(6)),
                  qml.expval(qml.PauliX(2) @ qml.PauliY(6)), qml.expval(qml.PauliY(2) @ qml.PauliZ(6)),
                  qml.expval(qml.PauliZ(2) @ qml.PauliX(6))]
    elif latent_dim == 15:
        result = [qml.expval(qml.PauliX(2)), qml.expval(qml.PauliY(2)), qml.expval(qml.PauliZ(2)),
                  qml.expval(qml.PauliX(6)), qml.expval(qml.PauliY(6)), qml.expval(qml.PauliZ(6)),
                  qml.expval(qml.PauliX(2) @ qml.PauliX(6)), qml.expval(qml.PauliY(2) @ qml.PauliY(6)),
                  qml.expval(qml.PauliZ(2) @ qml.PauliZ(6)),
                  qml.expval(qml.PauliX(2) @ qml.PauliY(6)), qml.expval(qml.PauliY(2) @ qml.PauliZ(6)),
                  qml.expval(qml.PauliZ(2) @ qml.PauliX(6)),
                  qml.expval(qml.PauliX(2) @ qml.PauliZ(6)), qml.expval(qml.PauliY(2) @ qml.PauliX(6)),
                  qml.expval(qml.PauliZ(2) @ qml.PauliY(6))]

    return result

def U_SU4(params, wires): # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


def conv_layer_1(U, params):
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])
    U(params, wires=[7, 0])


def conv_layer_2(U, params):
    U(params, wires=[2, 4])
    U(params, wires=[6, 0])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])


def conv_layer_3(U, params):
    U(params, wires=[2, 6])
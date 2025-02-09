import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 8
dev = qml.device("lightning.qubit", wires=n_qubits)

class QSVDDModel(nn.Module):
    def __init__(self, n_layers, n_qubits, num_params_conv):
        super().__init__()
        self.quantum_weights = nn.Parameter(torch.randn((n_layers*num_params_conv), dtype=torch.float32))
        print("Weights", self.quantum_weights.shape)
        self.n_qubits = n_qubits
        self.num_params_conv = num_params_conv

    def forward(self, x):
        b_s = x.shape[0]
        trash_measurements = []

        for i in range(b_s):
            measurements = quantum_circuit(x[i].view(-1), self.quantum_weights, self.n_qubits, self.num_params_conv)
            trash_measurements.append(torch.stack(measurements))

        return torch.stack(trash_measurements)


@qml.qnode(dev, interface="torch") # , diff_method = 'backprop'
def quantum_circuit(inputs, weights, n_qubits, num_params_conv):

    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)

    conv_layer_1(U_SU4, weights[0:num_params_conv])
    conv_layer_1(U_SU4, weights[num_params_conv: 2 * num_params_conv])
    conv_layer_2(U_SU4, weights[2 * num_params_conv: 3 * num_params_conv])
    conv_layer_2(U_SU4, weights[3 * num_params_conv: 4 * num_params_conv])
    conv_layer_3(U_SU4, weights[4 * num_params_conv: 5 * num_params_conv])

    # Measurements
    result = [qml.expval(qml.PauliX(2)), qml.expval(qml.PauliY(2)), qml.expval(qml.PauliZ(2)),
              qml.expval(qml.PauliX(6)), qml.expval(qml.PauliY(6)), qml.expval(qml.PauliZ(6)),
              qml.expval(qml.PauliX(2) @ qml.PauliX(6)), qml.expval(qml.PauliY(2) @ qml.PauliY(6)),
              qml.expval(qml.PauliZ(2) @ qml.PauliZ(6))]

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
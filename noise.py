import pennylane as qml
import pennylane.noise as qnoise
import matplotlib.pyplot as plt

single_qubit_cond = qml.noise.op_in(['X', 'Y', 'Z', 'Hadamard', 'I', 'RX', 'RY', 'RZ', 'U1', 'U2', 'U3'])
two_qubit_cond = qml.noise.op_in(['CNOT', 'CZ', 'SWAP', 'ISWAP', 'CPhase', 'CRX', 'CRY', 'CRZ'])

IBM_KYIV_SX_ERROR = 2.691e-4
IBM_KYIV_CX_ERROR = 1.57e-2
IBM_KYIV_T1 = 278.79
IBM_KYIV_T2 = 111.79
IBM_KYIV_TG_1 = 32
IBM_KYIV_TG_2 = 84

def single_qubit_depolarizing(op, **kwargs):
    """Apply a depolarizing error to a single qubit gate."""
    return qml.DepolarizingChannel(IBM_KYIV_SX_ERROR, wires=op.wires[0])

def two_qubit_depolarizing(op, **kwargs):
    """Apply a depolarizing error to a two qubit gate."""
    return qml.DepolarizingChannel(IBM_KYIV_CX_ERROR, wires=op.wires[-1])


def one_qubit_thermal_relaxation(op, **kwargs):
    """Apply thermal relaxation error to a single qubit gate."""
    return qml.ThermalRelaxationError(0, IBM_KYIV_T1, IBM_KYIV_T2, IBM_KYIV_TG_1, wires=op.wires[0])


def two_qubit_thermal_relaxation(op, **kwargs):
    """Apply thermal relaxation error to a two qubit gate."""
    return qml.ThermalRelaxationError(0, IBM_KYIV_T1, IBM_KYIV_T2, IBM_KYIV_TG_2, wires=op.wires[-1])

def noise_fn(op, **kwargs):
    one_qubit_thermal_relaxation(op, **kwargs)
    single_qubit_depolarizing(op, **kwargs)

def noise_fn_2(op, **kwargs):
    two_qubit_depolarizing(op, **kwargs)
    two_qubit_thermal_relaxation(op, **kwargs)

def noise_model():
    nm = qnoise.NoiseModel({
        single_qubit_cond: noise_fn,
        two_qubit_cond: noise_fn_2
    })
    return nm

@qml.qnode(qml.device('default.mixed', wires=2))
def noisy_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs()


if __name__ == '__main__':
    noise_model = noise_model()
    node = qml.add_noise(noisy_circuit, noise_model=noise_model)
    print(node())
    qml.draw_mpl(noisy_circuit)()

    plt.show()



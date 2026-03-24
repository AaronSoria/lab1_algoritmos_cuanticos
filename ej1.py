import pennylane as qml
from pennylane import numpy as np

# Dispositivo de 3 qubits
dev = qml.device("default.qubit", wires=3)

# =========================
# Circuito propuesto (tu solución)
# =========================

@qml.qnode(dev)
def circuito_propuesto():
    
    # H en qubit objetivo (2)
    qml.Hadamard(wires=2)
    
    # Bloque tipo Toffoli descompuesto
    qml.CNOT(wires=[1,2])
    qml.T(wires=2)
    
    qml.CNOT(wires=[0,2])
    qml.adjoint(qml.T)(wires=2)
    
    qml.CNOT(wires=[1,2])
    qml.T(wires=2)
    
    qml.CNOT(wires=[0,2])
    qml.adjoint(qml.T)(wires=2)
    
    qml.T(wires=1)
    qml.CNOT(wires=[0,1])
    
    qml.adjoint(qml.T)(wires=1)
    qml.CNOT(wires=[0,1])
    
    qml.T(wires=0)
    
    # H final
    qml.Hadamard(wires=2)
    
    return qml.state()

# =========================
# Toffoli exacto
# =========================

@qml.qnode(dev)
def toffoli_exacto():
    qml.Toffoli(wires=[0,1,2])
    return qml.state()

# =========================
# Comparación
# =========================

estado_propuesto = circuito_propuesto()
estado_exacto = toffoli_exacto()

fidelidad = np.abs(np.vdot(estado_propuesto, estado_exacto))**2

print("Estado propuesto:\n", estado_propuesto)
print("\nEstado exacto:\n", estado_exacto)
print(f"\nFidelidad: {fidelidad:.10f}")
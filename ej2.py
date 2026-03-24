import pennylane as qml
from pennylane import numpy as np

# =========================
# Dispositivo de 3 qubits
# =========================
dev = qml.device("default.qubit", wires=3)

# =========================
# Estado objetivo
# =========================
phi = (1/np.sqrt(7)) * np.array([0,1,1,1,1,1,1,1], dtype=complex)

# =========================
# Ansatz variacional
# =========================
@qml.qnode(dev)
def circuito(theta):
    
    # Primera capa de rotaciones
    qml.RY(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    qml.RY(theta[2], wires=2)
    
    # Entrelazamiento
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[0,2])
    
    # Segunda capa
    qml.RY(theta[3], wires=0)
    qml.RY(theta[4], wires=1)
    qml.RY(theta[5], wires=2)
    
    # Entrelazamiento
    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[0,2])
    
    # Tercera capa
    qml.RY(theta[6], wires=0)
    qml.RY(theta[7], wires=1)
    qml.RY(theta[8], wires=2)
    
    return qml.state()

# =========================
# Función de coste
# =========================
def coste(theta):
    psi = circuito(theta)
    overlap = np.sum(np.conj(phi) * psi)
    return 1 - np.abs(overlap)**2

# =========================
# Inicialización
# =========================
np.random.seed(42)
theta = np.random.normal(0, 0.1, 9)

# Optimizador
opt = qml.GradientDescentOptimizer(stepsize=0.05)

# =========================
# Entrenamiento
# =========================
for i in range(1000):
    theta = opt.step(coste, theta)

# =========================
# Resultados
# =========================
psi_final = circuito(theta)
fidelidad = np.abs(np.vdot(phi, psi_final))**2

print("\nFidelidad final:", fidelidad)

# Probabilidades
probs = np.abs(psi_final)**2

print("\nDistribución final:")
for i, p in enumerate(probs):
    print(f"|{format(i, '03b')}>: {p:.6f}")
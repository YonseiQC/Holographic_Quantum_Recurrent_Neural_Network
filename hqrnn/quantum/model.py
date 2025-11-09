import pennylane as qml
import jax.numpy as jnp
from jax import random
from hqrnn.config.base import Config

# --- 7. Quantum Model

class QuantumModel:
    def __init__(self, config: Config):
        self.config = config
        self.wires = list(range(config.model_cfg.n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires, shots=None)  # Analytic mode for probs

        
        # ------ QNodes

        @qml.qnode(self.dev, interface="jax")
        def circuit_for_probs(params, h_state, x_row, label_idx, use_x: bool = True):
            self._circuit_content(params, h_state, x_row, label_idx, use_x)
            return qml.probs(wires=range(self.config.model_cfg.n_D))  # Measure data register qubits

        @qml.qnode(self.dev, interface="jax")
        def circuit_for_state(params, h_state, x_row, label_idx, use_x: bool = True):
            self._circuit_content(params, h_state, x_row, label_idx, use_x)
            return qml.state()  # Full statevector

        self.circuit_for_probs = circuit_for_probs
        self.circuit_for_state = circuit_for_state


    # ------ Circuit content

    def _circuit_content(self, params, h_state, x_row, label_idx, use_x):
        cfg = self.config.model_cfg
        qml.StatePrep(h_state, wires=range(cfg.n_D, cfg.n_qubits))  # Prepare hidden register state
        for i in range(cfg.n_H):
            qml.RZ(params["label_embed"][label_idx, i], wires=cfg.n_D + i)  # Label embedding on hidden qubits (model3)
        if use_x:
            for i in range(cfg.n_D):
                qml.RY(jnp.pi * x_row[i], wires=i)  # Encode input bits on data qubits
        self.ansatz(params, self.wires)  # Ansatz


    # ------ Parameter initialization

    def initialize_params(self, key, n_classes=2):
        cfg = self.config.model_cfg
        keys = random.split(key, 5)
        theta_1q = random.normal(keys[0], (cfg.depth, cfg.n_qubits, 3)) * jnp.sqrt(2.0 / (cfg.n_qubits * 3))  # U3 angles
        theta_zzD = random.normal(keys[1], (cfg.depth, cfg.n_D)) * jnp.sqrt(2.0 / cfg.depth)  # ZZ on data ring
        theta_zzH = random.normal(keys[2], (cfg.depth, cfg.n_H)) * jnp.sqrt(2.0 / cfg.depth)  # ZZ on hidden ring
        theta_zzX = random.normal(keys[3], (min(cfg.n_D, cfg.n_H),)) * jnp.sqrt(2.0 / cfg.depth)  # ZZ across registers
        label_embed = random.normal(keys[4], (n_classes, cfg.n_H)) * 0.1  # Class embedding on hidden
        return {
            "theta_1q": theta_1q, "theta_zzD": theta_zzD,
            "theta_zzH": theta_zzH, "theta_zzX": theta_zzX,
            "label_embed": label_embed
        }


    # ------ Ansatz

    def ansatz(self, params, wires):
        cfg = self.config.model_cfg
        for d in range(cfg.depth):
            for i in range(cfg.n_qubits):
                qml.U3(params["theta_1q"][d, i, 0], params["theta_1q"][d, i, 1], params["theta_1q"][d, i, 2], wires=wires[i])  # Per-qubit SU(2)
            for i in range(cfg.n_D):
                qml.IsingZZ(params["theta_zzD"][d, i], wires=[i, (i + 1) % cfg.n_D])  # Data ring entanglement
            for i in range(cfg.n_H):
                qml.IsingZZ(params["theta_zzH"][d, i], wires=[cfg.n_D + i, cfg.n_D + (i + 1) % cfg.n_H])  # Hidden ring entanglement
            for i in range(min(cfg.n_D, cfg.n_H)):
                qml.IsingZZ(params["theta_zzX"][i], wires=[i, cfg.n_D + i])  # Data–hidden coupling  # param-count hint: 4*depth*n_qubits + n_D + n_classes*n_H

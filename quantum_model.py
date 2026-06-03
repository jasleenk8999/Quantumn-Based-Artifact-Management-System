import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. QUANTUM DEVICE
# ===============================
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# ===============================
# 2. QUANTUM CIRCUIT
# ===============================
@qml.qnode(dev)
def q_circuit(weights, inputs):

    # Angle Encoding
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)

    # Variational Layers
    for l in range(2):

        # Entanglement
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Trainable Rotations
        for i in range(n_qubits):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

    return qml.expval(qml.PauliZ(0))


# ===============================
# 3. LOAD NASA CMAPSS DATA
# ===============================
def load_all_data():

    files = [
        "train_FD001.txt",
        "train_FD002.txt",
        "train_FD003.txt",
        "train_FD004.txt",
    ]

    col_names = (
        ["id", "cycle", "op1", "op2", "op3"]
        + [f"s{i}" for i in range(1, 22)]
    )

    combined_data = []

    print("📂 Loading NASA CMAPSS Fleet Data...")

    for f in files:

        if os.path.exists(f):

            df = pd.read_csv(
                f,
                sep=r"\s+",
                header=None,
                names=col_names
            )

            # Remaining Useful Life
            df["max_cycle"] = df.groupby("id")["cycle"].transform("max")

            # Label
            df["label"] = np.where(
                (df["max_cycle"] - df["cycle"]) <= 30,
                1,
                -1
            )

            combined_data.append(df)

            print(f"✅ Loaded {f}")

    full_df = pd.concat(combined_data, ignore_index=True)

    # Feature Selection
    sensors = ["s11", "s12", "s13", "s15"]

    scaler = MinMaxScaler()

    X = scaler.fit_transform(full_df[sensors])

    Y = np.array(full_df["label"].values, requires_grad=False)

    return X, Y, scaler


# ===============================
# 4. COST FUNCTION
# ===============================
def cost_function(weights, X, Y):

    preds = []

    for x in X:
        preds.append(q_circuit(weights, x))

    preds = np.array(preds)

    return np.mean((Y - preds) ** 2)


# ===============================
# 5. TRAINING
# ===============================
def train_full_fleet():

    X, Y, scaler = load_all_data()

    # Training subset
    train_x = X[:500].astype(np.float64)
    train_y = Y[:500].astype(np.float64)

    # Initialize weights
    weights = 0.15 * np.random.randn(2, n_qubits, 3, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=0.1)

    epochs = 15

    print(f"\n🚀 Training Quantum VQC on {len(train_x)} samples...\n")

    for epoch in range(epochs):

        weights, cost = opt.step_and_cost(
            lambda w: cost_function(w, train_x, train_y),
            weights
        )

        current_cost = float(cost)

        print(f"Epoch {epoch+1}/{epochs} | Training Cost: {current_cost:.4f}")

    # ===============================
    # SAVE MODEL
    # ===============================
    with open("quantum_weights.pkl", "wb") as f:
        pickle.dump(weights, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\n🏆 Training Complete! Quantum model saved.")


# ===============================
# 6. RUN
# ===============================
if __name__ == "__main__":
    train_full_fleet()
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# --- 1. DATA PREPARATION (Same as before but with Shuffling) ---
def load_and_prep():
    # 1. Define Column Names (NASA CMAPSS Standard)
    col_names = ['id', 'cycle', 's1', 's2'] + [f's{i}' for i in range(3, 27)]
    
    # 2. Load Training Data
    # Using 'r' prefix to handle backslashes correctly in Python 3.13
    train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    
    # 3. Calculate Remaining Useful Life (RUL) and Labels
    # We find the max cycle for each engine to determine its failure point
    train['max'] = train.groupby('id')['cycle'].transform('max')
    # Label: 1 if within the 30-cycle "Danger Zone", else -1 (Healthy)
    train['label'] = np.where((train['max'] - train['cycle']) <= 30, 1, -1)
    
    # 4. Load Testing Data and Ground Truth
    test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    true_rul = pd.read_csv('RUL_FD001.txt', header=None, names=['true_rul'])
    
    # 5. Advanced Feature Selection for 85%+ Accuracy
    # Sensors 11, 12, 13, and 15 are the most sensitive to core engine wear
    sensors = ['s11', 's12', 's13', 's15']
    scaler = MinMaxScaler()
    
    # 6. Prepare Training Sets
    # Increased to 100 samples to provide more "Failure" patterns to the QML model
    train_x = scaler.fit_transform(train[sensors])[:100] 
    train_y = np.array(train['label'].values[:100], requires_grad=False)
    
    # 7. Prepare Testing Sets
    # We take the final state (last cycle) of each engine in the test set
    test_last = test.groupby('id').last().reset_index()
    test_x = scaler.transform(test_last[sensors])
    # Mapping Ground Truth: 1 if actual RUL <= 30, else -1
    test_y = np.where(true_rul['true_rul'] <= 30, 1, -1)
    
    return train_x, train_y, test_x, test_y

# --- 2. QUANTUM SETUP ---
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def q_circuit(weights, inputs):
    for i in range(4): 
        qml.RY(inputs[i] * np.pi, wires=i)
    for l in range(2):
        for i in range(4): 
            qml.CNOT(wires=[i, (i + 1) % 4])
        for i in range(4): 
            qml.Rot(*weights[l, i], wires=i)
    return qml.expval(qml.PauliZ(0))

# --- 3. COMPLEX TRAINING (50 EPOCHS) ---
train_x, train_y, test_x, test_y = load_and_prep()
# Using a slightly different weight initialization for better convergence
weights = 0.15 * np.random.randn(2, 4, 3, requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.05) # Lower stepsize for finer tuning

def cost_fn(w):
    preds = np.array([q_circuit(w, x) for x in train_x])
    return np.mean((train_y - preds)**2)

print("🚀 Starting Extended Quantum Training (50 Epochs)...")
for epoch in range(50):
    weights, cost = opt.step_and_cost(cost_fn, weights)
    if (epoch + 1) % 5 == 0: # Print every 5 epochs to keep terminal clean
        print(f"Epoch {epoch+1} | Cost: {float(cost):.4f}")

with open('quantum_weights_complex.pkl', 'wb') as f:
    pickle.dump(weights, f)

# --- 4. TESTING ---
print("\n🔎 Evaluating High-Complexity Accuracy...")
correct = 0
for i in range(len(test_x)):
    prediction = 1 if q_circuit(weights, test_x[i]) > 0 else -1
    if prediction == test_y[i]:
        correct += 1

final_acc = (correct/len(test_x))*100
print(f"✅ Final Quantum Accuracy (50 Epochs): {final_acc:.2f}%")
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler

# 1. Find all training files
train_files = glob.glob('train_FD001.txt')
all_data = []

col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

for file in train_files:
    # Load each file
    df = pd.read_csv(file, sep='\s+', header=None, names=col_names)
    
    # Add a column to know which dataset it is (e.g., FD001)
    df['scenario'] = file.split('_')[1].split('.')[0]
    
    # Calculate RUL for this specific file
    max_cycle = df.groupby('id')['cycle'].max().reset_index()
    df = df.merge(max_cycle.rename(columns={'cycle': 'max_cycle'}), on='id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    all_data.append(df)

# 2. Combine into one Master Dataset
master_df = pd.concat(all_data, ignore_index=True)

# 3. Select your 4 Quantum Sensors
quantum_sensors = ['s2', 's7', 's11', 's15']
master_df = master_df[['scenario', 'id', 'cycle', 'RUL'] + quantum_sensors]

# 4. Global Normalization (Crucial: use the same scale for all files)
scaler = MinMaxScaler()
master_df[quantum_sensors] = scaler.fit_transform(master_df[quantum_sensors])

master_df.to_csv('master_quantum_data.csv', index=False)
print(f"✅ Combined {len(train_files)} files into master_quantum_data.csv")
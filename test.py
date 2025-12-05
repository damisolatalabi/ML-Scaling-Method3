# debug_test.py
import numpy as np
import pickle
import os
from model import HMM

def load_sequence(filepath):
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                line = line.strip().replace('(', '').replace(')', '')
                x, y, z = map(float, line.split(','))
                points.append([x, y, z])
    return np.array(points) / 10.0

# Load models
models = {}
for file in os.listdir('model_parameters3'):
    if file.endswith('.pkl'):
        with open(f'model_parameters3/{file}', 'rb') as f:
            model_data = pickle.load(f)
            models[model_data['label']] = model_data

# Test one sequence and print ALL probabilities
test_file = "clean_data/circle/1.txt"
sequence = load_sequence(test_file)

print(f"Testing: {test_file}")
print(f"Sequence shape: {sequence.shape}")
print(f"First point: {sequence[0]}")

print("\nProbabilities from each model:")
for label, model_data in models.items():
    hmm = HMM(3, label, velocity=False)
    hmm.PI = model_data['PI']
    hmm.A = model_data['A']
    hmm.mu = model_data['mu']
    hmm.var = model_data['var']
    
    alpha = hmm.forward(sequence)
    prob = np.sum(alpha[-1])
    print(f"{label}: {prob:.10f}")

# Check if models have different parameters
print("\nModel parameter comparison:")
for label, model_data in models.items():
    print(f"\n{label}:")
    print(f"  PI: {model_data['PI']}")
    print(f"  mu shape: {model_data['mu'].shape}")
    print(f"  mu first row: {model_data['mu'][0] if model_data['mu'].ndim > 1 else model_data['mu']}")
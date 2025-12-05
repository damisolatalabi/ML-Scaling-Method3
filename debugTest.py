# debug_train.py
import numpy as np
from model import HMM

# Create a simple test sequence
test_seq = np.array([
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4], 
    [0.3, 0.4, 0.5]
])

print("Creating HMM...")
hmm = HMM(3, 'test', velocity=False)

print("\nTraining with simple data...")
hmm.train([test_seq], max_iterations=5)

print("\nModel parameters after training:")
print(f"PI: {hmm.PI}")
print(f"mu shape: {hmm.mu.shape}")
print(f"mu: {hmm.mu}")
print(f"var shape: {hmm.var.shape}")
print(f"var first: {hmm.var[0]}")

# Test forward
print("\nTesting forward...")
alpha = hmm.forward(test_seq)
log_prob = np.sum(alpha[-1])
print(f"Log probability: {log_prob}")
print(f"Probability: {np.exp(log_prob)}")
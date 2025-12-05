import model
import numpy as np
import os
import time
from multiprocessing import Process
import random

def prepare_training_set(type, source):
    folder = f'{source}/{type}'
    samples = os.listdir(folder)

    train_sequences = []

    for sample in samples:
        sequence = []
    
        f = open(os.path.join(folder,sample))
        text = f.read()

        text = text.split('\n')

        for point in text:
            if len(point) < 2:
                continue

            point = point.replace('(',"")
            point = point.replace(')',"")

            point = point.split(',')
            point[0] = float(point[0])
            point[1] = float(point[1])
            point[2] = float(point[2])

            sequence.append(point)
        
        # Skip empty sequences to avoid IndexError
        if len(sequence) > 0:
            # Scale down by 10
            sequence = np.array(sequence) / 10.0
            train_sequences.append(sequence)
        else:
            print(f"Warning: Empty file {sample}, skipping")

    return train_sequences

def model_info(model):
    info = model.model_info()
    print(f"Model : {info[0]}")
    print(f"Hidden states : {info[1]}")
    print(f"PI : {info[2]}")
    print(f"A : {info[3]}")
    print(f"Mean : {info[4]}")
    print(f"Variance : {info[5]}")

def train(model, set):
    model.train(set) 

# Initialize models + number of hidden states
hidden_states = 3
source_training = 'augmented_data_Method3'

model_set = [
    model.HMM(hidden_states, 'circle', velocity=False),
    model.HMM(hidden_states, 'diagonal_left', velocity=False),
    model.HMM(hidden_states, 'diagonal_right', velocity=False),
    model.HMM(hidden_states, 'horizontal', velocity=False),
    model.HMM(hidden_states, 'vertical', velocity=False)
]

# Check if source folder exists
if not os.path.exists(source_training):
    print(f"ERROR: Folder '{source_training}' not found!")
    print("Available folders:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  - {item}")
    exit(1)

# Check each class folder
classes = ['circle', 'vertical', 'diagonal_left', 'diagonal_right', 'horizontal']
for cls in classes:
    class_path = f'{source_training}/{cls}'
    if not os.path.exists(class_path):
        print(f"WARNING: Class folder '{class_path}' not found")
    else:
        file_count = len([f for f in os.listdir(class_path) if f.endswith('.txt')])
        print(f"Found {file_count} files in {class_path}")

training_sets = [
    prepare_training_set('circle', source_training),
    prepare_training_set('diagonal_left', source_training),
    prepare_training_set('diagonal_right', source_training),
    prepare_training_set('horizontal', source_training),
    prepare_training_set('vertical', source_training)
]

# Check if we have any training data
total_sequences = sum(len(s) for s in training_sets)
print(f"\nTotal training sequences: {total_sequences}")

if total_sequences == 0:
    print("ERROR: No training data found!")
    exit(1)

# train models
start = time.time()

for model_obj, train_set in zip(model_set, training_sets):
    if len(train_set) == 0:
        print(f"Skipping {model_obj.get_label()} - no training data")
        continue
        
    print(f"Training {model_obj.get_label()} with {len(train_set)} sequences")
    try:
        model_obj.train(train_set)
    except Exception as e:
        print(f"Error training {model_obj.get_label()}: {e}")

end = time.time()

print(f"\nTraining finished in {end - start:.2f}s")

os.makedirs(f'model_parameters{hidden_states}', exist_ok=True)

# save model parameters
for model in model_set:
    model.save(f'model_parameters{hidden_states}/{model.get_label()}.pkl')
    print(f"Saved model: {model.get_label()}.pkl")

print(f"\nModels saved to: model_parameters{hidden_states}/")
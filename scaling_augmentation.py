import os
import random

def scale_point_cloud(points, scale_factor):
    """Scale all points by the same factor"""
    return [(x * scale_factor, y * scale_factor, z * scale_factor) for x, y, z in points]

def read_points_from_file(filename):
    """Read points from a text file"""
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line[0] == '(':  # Check if it's a coordinate line
                # Remove parentheses and split by commas
                coords = line[1:-1].split(',')
                points.append((float(coords[0]), float(coords[1]), float(coords[2])))
    return points

def write_points_to_file(points, filename):
    """Write points to a text file"""
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"({point[0]},{point[1]},{point[2]})\n")

def augment_class_data(class_folder, output_folder, target_count=200):
    """Augment data for one class to reach target count"""
    # Get all original files
    original_files = [f for f in os.listdir(class_folder) if f.endswith('.txt')]
    
    # Calculate how many augmented versions we need per original file
    aug_per_file = target_count // len(original_files)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    augmented_count = 0
    
    for orig_file in original_files:
        # Read original points
        orig_points = read_points_from_file(os.path.join(class_folder, orig_file))
        
        # Create multiple scaled versions
        for i in range(aug_per_file):
            # Generate random scale factor between 0.9 and 1.1
            scale_factor = random.uniform(0.9, 1.1)
            
            # Scale the points
            scaled_points = scale_point_cloud(orig_points, scale_factor)
            
            # Create new filename
            new_filename = f"{orig_file[:-4]}_scaled_{i+1}.txt"
            
            # Save scaled points
            write_points_to_file(scaled_points, os.path.join(output_folder, new_filename))
            augmented_count += 1
    
    print(f"Created {augmented_count} augmented files for {class_folder}")
    return augmented_count

# Main execution
if __name__ == "__main__":
    # Define your class folders - USE EXACT FOLDER NAMES
    classes = ['circle', 'vertical', 'diagonal_left', 'diagonal_right', 'horizontal']
    
    for class_name in classes:
        input_folder = f"clean_data/{class_name}"  # Path to your clean data
        output_folder = f"augmented_data_Method3/{class_name}"  # Where to save augmented data
        
        augment_class_data(input_folder, output_folder, target_count=200)
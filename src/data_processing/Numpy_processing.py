import numpy as np
import os
import json

# Define the directory paths
data_dir = '/home/abhishek/svg-model-project/data/processed'
output_dir = '/home/abhishek/svg-model-project/data_processing/processed_data'

# Define the number of M and Q points required for consistency
required_m_points = 31  # Example, adjust as needed
required_q_points = 31  # Example, adjust as needed

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Helper function to pad or trim points to the required length
def adjust_points(points, required_length):
    if len(points) > required_length:
        return points[:required_length]
    elif len(points) < required_length:
        padding = np.zeros((required_length - len(points), points.shape[1]))
        return np.vstack((points, padding))
    return points

# Process each class
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        m_points_list = []
        q_points_list = []
        stroke_widths_list = []
        
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(class_dir, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                m_points = np.array(data['m_points'])
                q_points = np.array(data['q_points'])
                stroke_widths = np.array(data['stroke_widths'])
                
                # Adjust points to the required shape
                m_points = adjust_points(m_points, required_m_points)
                q_points = adjust_points(q_points, required_q_points)
                
                m_points_list.append(m_points)
                q_points_list.append(q_points)
                stroke_widths_list.append(stroke_widths)
        
        if m_points_list and q_points_list and stroke_widths_list:
            # Convert lists to NumPy arrays
            m_points_array = np.array(m_points_list)
            q_points_array = np.array(q_points_list)
            stroke_widths_array = np.array(stroke_widths_list)
            
            # Save the processed data
            m_points_path = os.path.join(output_dir, f"{class_name}_m_points.npy")
            q_points_path = os.path.join(output_dir, f"{class_name}_q_points.npy")
            stroke_widths_path = os.path.join(output_dir, f"{class_name}_stroke_widths.npy")
            
            np.save(m_points_path, m_points_array)
            np.save(q_points_path, q_points_array)
            np.save(stroke_widths_path, stroke_widths_array)
            
            # Ensure files are not empty
            if (os.path.getsize(m_points_path) == 0 or 
                os.path.getsize(q_points_path) == 0 or 
                os.path.getsize(stroke_widths_path) == 0):
                print(f"Error: One or more files for class '{class_name}' are empty.")
            else:
                print(f"Processed data for class: {class_name}")
        else:
            print(f"No valid data found for class: {class_name}")

print("Data processing complete.")

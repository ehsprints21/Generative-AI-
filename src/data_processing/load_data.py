import json
import numpy as np

def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    m_points_all = []
    q_points_all = []
    stroke_widths_all = []
    classes = []

    for entry in data:
        class_name = entry['class']
        classes.append(class_name)
        images_data = entry['images']
        
        for image_data in images_data:
            m_points = np.array(image_data['m_points'])
            q_points = np.array(image_data['q_points'])
            stroke_widths = np.array(image_data['stroke_widths'])
            
            m_points_all.append(m_points)
            q_points_all.append(q_points)
            stroke_widths_all.append(stroke_widths)
    
    return classes, m_points_all, q_points_all, stroke_widths_all

# Example usage
if __name__ == "__main__":
    json_file = '/home/abhishek/svg-model-project/data/processed/extracted_features.json'  # Replace with your actual JSON file path
    data = load_json_data(json_file)
    classes, m_points, q_points, stroke_widths = preprocess_data(data)
    
    # Example of accessing the processed data
    print(f"Classes: {classes}")
    print(f"Number of images: {len(m_points)}")
    print(f"Example m_points for first image: {m_points[0]}")
    print(f"Example q_points for first image: {q_points[0]}")
    print(f"Example stroke_widths for first image: {stroke_widths[0]}")

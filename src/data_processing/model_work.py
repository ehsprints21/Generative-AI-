import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from svgpathtools import Path, svg2paths

# Load the saved model
autoencoder = load_model('/path/to/save/autoencoder_model.h5')

# Function to predict M points, Q points, and stroke widths based on class
def predict_and_generate_svg(class_num):
    # Load data based on class_num
    json_file = f'/path/to/class_{class_num}_data.json'  # Adjust path and filename accordingly
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Preprocess data (similar to training preprocessing)
    X_m_points = np.array([item['m_points'] for item in data])
    X_q_points = np.array([item['q_points'] for item in data])
    X_stroke_widths = np.array([item['stroke_widths'] for item in data])

    # Predict using the autoencoder
    predicted_m_points, predicted_q_points, predicted_stroke_widths = autoencoder.predict([X_m_points, X_q_points, X_stroke_widths])

    # Generate SVG file from predicted data
    svg_paths = []
    for m_points, q_points, stroke_widths in zip(predicted_m_points, predicted_q_points, predicted_stroke_widths):
        # Assuming you have a function to convert points to SVG Path objects
        path = points_to_svg_path(m_points, q_points, stroke_widths)
        svg_paths.append(path)

    # Combine all paths into an SVG file
    svg_file = f'/path/to/save/class_{class_num}_output.svg'
    with open(svg_file, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg">\n')
        for path in svg_paths:
            f.write(path.d())
        f.write('</svg>')

    print(f'SVG file saved: {svg_file}')

# Example usage
class_number = 7  # Replace with the desired class number
predict_and_generate_svg(class_number)

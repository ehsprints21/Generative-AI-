import os
import xml.etree.ElementTree as ET
import json
import re

def extract_features_from_svg(svg_path):
    m_points = []
    q_points = []
    stroke_widths = []

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Namespace for SVG elements
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        for path in root.findall('.//svg:path', ns):
            d_attr = path.get('d')
            stroke_width = float(path.get('stroke-width', 1))  # Default to 1 if not specified

            # Log d attribute for debugging
            print(f"Parsing path with d attribute: {d_attr}")

            # Parse M and Q points
            segments = re.findall(r'[MLQ]\s*[^MLQ]*', d_attr)
            for segment in segments:
                if segment.startswith('M'):
                    points = segment[1:].strip().split(',')
                    if len(points) >= 2:
                        try:
                            m_x = float(points[0].strip())
                            m_y = float(points[1].strip())
                            m_points.append([m_x, m_y])
                        except ValueError as e:
                            print(f"Error parsing M point: {e}")
                elif segment.startswith('Q'):
                    points = segment[1:].strip().split(',')
                    if len(points) >= 4:
                        try:
                            q_x1 = float(points[0].strip())
                            q_y1 = float(points[1].strip())
                            q_x2 = float(points[2].strip())
                            q_y2 = float(points[3].strip())
                            q_points.append([q_x1, q_y1, q_x2, q_y2])
                        except ValueError as e:
                            print(f"Error parsing Q point: {e}")

            stroke_widths.append(stroke_width)

    except ET.ParseError as e:
        print(f"Error parsing XML in {svg_path}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {svg_path}: {e}")

    return {
        'm_points': m_points,
        'q_points': q_points,
        'stroke_widths': stroke_widths
    }

# Main Processing Loop
root_dir = '/home/abhishek/svg-model-project/data/raw'
dataset = []

for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)
    if os.path.isdir(class_path):
        images_data = []
        for svg_file in os.listdir(class_path):
            if svg_file.endswith('.svg'):
                svg_file_path = os.path.join(class_path, svg_file)
                try:
                    features = extract_features_from_svg(svg_file_path)
                    images_data.append(features)
                except Exception as e:
                    print(f"Error processing {svg_file_path}: {e}")
        dataset.append({
            'class': class_folder,
            'images': images_data
        })

# Save the JSON output
output_file = '/home/abhishek/svg-model-project/src/data_processing/dataset.json'
with open(output_file, 'w') as outfile:
    json.dump(dataset, outfile, indent=4)

print(f"JSON data saved to {output_file}")

# SVG_Generation_Model

## Project Overview
This project focuses on extracting M and Q points from SVG files, processing this data, training a machine learning model, and serving predictions via an API. The aim is to build a robust system that can efficiently interpret and generate vector line art.

## Project Structure

- **data/**: Directory for storing data files.
- **models/**: Directory for saving trained models.
- **notebooks/**: Directory for Jupyter notebooks.
- **src/**: Source code directory.
  - **extract_svg_data.py**: Script for extracting data from SVG files.
  - **prepare_data.py**: Script for preparing the data for training.
  - **train_model.py**: Script for training the machine learning model.
  - **evaluate_model.py**: Script for evaluating the trained model.
  - **serve_model.py**: Script for serving the model in production.
- **requirements.txt**: File listing the project's dependencies.
- **README.md**: Project documentation.

## Current Status

We have successfully completed the data creation phase, which involved extracting and organizing M and Q points from the SVG files. The next steps include data preparation, model training, evaluation, and deployment.

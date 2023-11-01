# NITEX-AI-Challenge

# Sustainable Apparel Classifier

This project involves building a machine learning model to classify sustainable apparel products using the Fashion MNIST dataset.

## Dataset

[Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)


## Code Structure

The project follows a modular structure:
- `evaluate_model.py`: Script for evaluating the trained model on a given dataset folder.
- `fashion_model.h5`: Saved CNN model.
- `requirements.txt`: List of dependencies.

## Usage

1. **Environment Setup:**
   - Create a virtual environment: `python -m venv venv`
   - Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
   - Install dependencies: `pip install -r requirements.txt`

2. **Model Evaluation:**
   - Run the evaluation script: `python evaluate_model.py /path/to/dataset/folder`

3. **Output Generation:**
   - The script generates an `output.txt` file with model details and evaluation metrics.

4. **Note:**
   - Ensure TensorFlow and other dependencies are correctly installed.
   - Modify the paths and filenames as needed.


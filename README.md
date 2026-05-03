# Loan Repayment QuAM: Predicting Customer Defaults & Repayment Latency

### Authors
Junaid Yusuf (jyusuf) and Hussain Bin Usman (hussainu)

## Project Description

Project Name: Loan Repayment QuAM

The Loan Repayment QuAM is a machine learning interface made to predict loan repayment outcomes using the final prepared Mexico dataset from the project. The interface allows the user to load labelled customer data, load a separate unlabelled query data file, train selected machine learning models, adjust important model parameters, and use inference to generate predictions for customer records (of the query file).

The software supports two main prediction tasks:
- Classification: predicting customer default / delinquency risk using `default_flag`
- Regression: predicting repayment latency using `repayment_months`

For classification, the final model used is a Kernel SVM. For regression, the final model used is a Decision Tree Regressor. These decisions are based on the comprehensive iteration testing that were made under `QuAM_report`.

## Submission Contents

The submitted folder contains the following files:

- `main.py`: main file used to launch the QuAM interface
- `interface.py`: contains the main Tkinter interface and user interaction
- `interface_helpers.py`: contains some helper functions for the interface (mainly styling, scrolling, and input validation)
- `models.py`: contains the actual ML models: preprocessing, model training, prediction, and inference logic
- `configurations.py`: contains configurations for the program storeed as constants (eg. colors, input ranges etc)
- `README.md`: this file
- `quam_test_files/`: folder containing example training and query CSV files


## Overview - How To Use the Program

### Required Packages
The project uses the following Python packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `tkinter`

### Summary of Usage Steps
Below is a summary of the steps required to use the interface:
1. Load a labelled training dataset.
2. Optionally load a separate unlabelled query dataset. If a query dataset is loaded, inference will use that dataset. Otherwise it'll use the training dataset.
3. Choose the prediction task and adjust the custom model parameters using the control panel on the left.
4. Train the selected model.
5. Enter a customer row index and generate an inference result. Alternatively, for classification, 'Locate All Predicted Delinquencies'


## Program Features Guide

### Dataset Inputs

### Custom Parameter Configuration

### Model Training

### Inference


## Example Test Files
The `quam_test_files` folder contains:

- `Training1Small.csv` and `Query1Small.csv`
- `Training2Medium.csv` and `Query2Medium.csv`
- `Training3Large.csv` and `Query3Large.csv`
- `Training4Balanced.csv` and `Query4Balanced.csv`
- `Training5FullDemo.csv` and `Query5FullDemo.csv`

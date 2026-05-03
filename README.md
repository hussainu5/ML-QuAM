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
The program allows for the uploading of two datasets, a lebelled training dataset and an unlabelled query dataset (optional). If a query dataset is loaded, inference will use that dataset. Otherwise it'll use the training dataset.

The datasets are uploaded by pressing 'Browse' which allows the user to upload the file after which they must Load the dataset.

### Custom Parameter Configuration

The custom parameter configuration section allows the user to choose the prediction task and adjust important model settings before training.

#### General parameters:
- `Prediction task`: chooses whether the program runs classification or regression.
- `Random seed`: controls reproducibility. Keeping the same seed helps produce the same train/evaluation split and results when the same data is used. Set as 15288 as a default.
- `Evaluation split size`: controls how much of the labelled training dataset is held out for evaluation when training the model. For example, `0.2` means 20% is used for evaluation and 80% is used for training.

#### Classification parameters:
- `Regularization parameter (C)`: controls how strongly the Kernel SVM penalizes classification mistakes. Larger values make the model fit the training data more strictly, while smaller values allow more flexibility.
- `Kernel influence (gamma)`: controls how much influence each training point has in the RBF kernel. The available options are `scale`, `auto`, and `fixed`.
- `Fixed gamma value`: appears only when `fixed` is selected for gamma. It allows the user to manually enter a positive gamma value.
- `Class balance weighting`: controls whether the model adjusts for class imbalance. `Balance classes (balanced)` gives more weight to underrepresented classes, while `Do not balance (none)` treats the classes equally.

#### Regression parameters:
- `Tree depth limit`: controls the maximum depth of the Decision Tree Regressor. Larger values allow a more complex tree, while smaller values keep the model simpler.
- `Minimum records per leaf`: controls the minimum number of records required in each final leaf of the decision tree. Larger values make the model smoother and help reduce overfitting to very small groups.

The interface checks all parameter inputs before training and shows an error message if an invalid value is entered.


### Model Training
Based on the parameters and the model selected, the model will be trained after pressing the 'Train Model' button'

### Inference
The user can enter a customer row index and generate an inference result (classification: predicted probability of delinquency for the customer; regression: predicted latency of repayment in months for the customer).

For classification, there is a special feature through the 'Locate All Predicted Delinquencies' button, which prints all the customers that are classed as delinquent (predicted to default with a probability >= 0.5). It displays the indexes of all thsee customers as well as their  respective probabilities of delinquency.

### Output Log
Displays the program's outputs to the user as text within the output log so they can track it.


## Example Test Files

### Overview
We prepared example input files for testing and demonstrating the final Loan Repayment QuAM interface. The files are generated from the final filtered Mexico dataset used throughout the project, so they stay consistent with the modelling work completed earlier. For each test case, a labelled training file and a separate unlabelled query file are created. The training file keeps the target columns needed to train the classification and regression models, while the paired query file does not include the labels by dropping the targets to imitate new customer records requiring inference. The training and query records in each pair are kept non-overlapping so that the interface is tested on unseen customer data rather than rows it has already trained on.

### Test Files Breakdown
Here is a breakdown of the example training and query files in `quam_test_files/:

- `Training1Small.csv` / `Query1Small.csv`: Small files for quick testing. The training file has **500 rows**, and the query file has **20 rows**.

- `Training2Medium.csv` / `Query2Medium.csv`: Medium-sized files for a more realistic but still manageable demonstration. The training file has **1,500 rows**, and the query file has **50 rows**.

- `Training3Large.csv` / `Query3Large.csv`: Larger files for testing the interface with more records. The training file has **3,000 rows**, and the query file has **100 rows**.

- `Training4Balanced.csv` / `Query4Balanced.csv`: Files designed to give the classification model a more balanced mix of delinquent and non-delinquent cases during training. The training file has **1,200 rows**, and the query file has **50 rows**.

- `Training5FullDemo.csv` / `Query5FullDemo.csv`: The largest demo files, used to test the interface in a fuller and more realistic setting. The training file has **5,000 rows**, and the query file has **150 rows**.

Each training file keeps the target columns needed for model training, while each paired query file removes target and leakage columns to imitate new customer records for inference.

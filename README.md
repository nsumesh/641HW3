# Comparative Analysis of RNN Architectures for Sentiment Classification

## Overview

This project evaluates the perfomance of three different RNN architectures for the task of sentiment classification on the IMDB dataset.

## Environment Setup, Python version ≥ 3.9

To set up the environment, run the following command to install the required dependencies
```plaintext
pip install -r requirements.txt
```
Make sure the directories exist 
```plaintext
mkdir -p data/preprocessed results
```
Place the IMDB dataset from Kaggle in the data directory.

## Process of running project

To first preprocess the dataset, run
```plaintext
python src/preprocess.py
```

Run the experiments for the different models and configurations
```plaintext
python src/train.py
```

The results are logged to the results directory after which a comparative analysis can be conducted to see how each model performed on different statistics such as Accuracy, F1 Score and the perfomance of the best and worst model.

A more detailed analysis of the model results and how it performed can be checked out at report.pdf at the directory root.



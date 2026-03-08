# Diabetes Dataset: EDA and Prediction  

This repository contains the project work for the **Programming and Database** course, part of the Data Science Master's degree at the University of Verona. The project focuses on the **Programming** section of the exam.

## Academic Information  
- **Course**: Programming and Database  
- **Professor**: Niccolò Marastoni  
- **Academic Year**: 2024/2025  

## Project Overview  
The goal of this project is to analyze and predict the likelihood of diabetes in patients based on health data. The dataset used is `diabetes_unclean.csv`, which contains raw health data from various patients.

### Objectives  
1. **Data Cleaning** — Handle missing values, inconsistencies, outliers, and encoding in the raw dataset.  
2. **Exploratory Data Analysis (EDA)** — Understand feature distributions, correlations, and key patterns.  
3. **Predictive Modeling** — Compare multiple classification algorithms (Logistic Regression, Decision Tree, KNN, Random Forest) on both balanced and imbalanced training sets.  
4. **Interactive Web Application** — A Streamlit app that walks through the full pipeline and allows real-time diabetes-likelihood predictions from user-supplied patient data.

## Dataset  
`datasets/diabetes_unclean.csv` — raw health records from various patients (1009 entries, 14 columns).  
Source: [Kaggle — Diabetes unclean dataset](https://www.kaggle.com/datasets/kabirolawalemohammed/diabetes-unclean)

## Repository Structure  
```
datasets/          Raw and cleaned datasets
notebooks/         Jupyter notebooks (EDA.ipynb, PRED.ipynb)
src/               Python scripts
  ST_EDA_prediction.py   Streamlit web application
  notebooks_export/      Exported notebook scripts
report/            Full LaTeX project report (PDF included)
abstract/          Original LaTeX course abstract
```

## Running the Streamlit App  
```bash
cd src
streamlit run ST_EDA_prediction.py
```

## Requirements  
Install all dependencies with:  
```bash
pip install -r requirements.txt
```

---
### Disclaimer  
This project has no scientific or medical validity. It is purely educational, developed to demonstrate data science techniques in the context of the **Programming and Database** course. Results must not be used for professional diagnosis or treatment.

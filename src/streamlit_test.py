import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt


#streamlit run streamlit_test.py



# Title and description
st.title("Diabetes Dataset: data-cleaning, EDA and Prediction")
st.markdown("""
This application showcase the cleaning methods and helps in performing an **Exploratory Data Analysis (EDA)** on the Diabetes dataset and 
provides predictions using various machine learning models.
""")


# Dataset Features Description
st.markdown("""
## Dataset features
In the dataset we can access the following feature:

- **Gender**: The gender of the individual (e.g., Male, Female).
- **AGE**: The age of the individual in years.
- **Urea**: The level of urea in the blood, indicating kidney function. 
- **Cr**: Creatinine level in the blood, used to assess kidney function.
- **HbA1c**: Hemoglobin A1c percentage, a measure of average blood sugar levels.
- **Chol**: Total cholesterol level in the blood, measured in mmol/L, indicating lipid profile.
- **TG**: Triglycerides level in the blood, measured in mmol/L, another component of the lipid profile.
- **HDL**: High-Density Lipoprotein cholesterol measured in mmol/L.
- **LDL**: Low-Density Lipoprotein cholesterol measured in mmol/L.
- **VLDL**: Very Low-Density Lipoprotein cholesterol measured in mmol/L.
- **BMI**: Body Mass Index, a measure of body fat based on height and weight.
- **CLASS**: The target variable indicating the presence or absence of diabetes.
""")

# Load default dataset
dataset = pd.read_csv("../datasets/diabetes_unclean.csv")
st.subheader("Dataset Overview")
st.write(dataset.head())
st.write(dataset.info())
st.write("Shape of the dataset:", dataset.shape)
st.write("Summary statistics:")
st.write(dataset.describe())

st.markdown("---")
# Data Cleaning Section
st.header("Data Cleaning")
st.markdown(
"""
## Data Cleaning and Preprocessing
In this section, data cleaning is performed by analyzing entries that are outside the acceptable range or are potential outliers. Additionally, preprocessing and encoding of non-numeric or categorical features are carried out.


List of task to be performed in this section:
- Backup the original dataset
- Check for Null values
- Conversion to numerical values or one-hot encoding of the 'object' or 'string' types
- Check for outliers and cleaning
""")

st.subheader("Backup the original dataset")
ds_backup = dataset.copy()
ds_backup.to_csv('../datasets/diabetes_unclean_backup.csv', index=False) # This saves a backup of the original dataset
st.write("Backup of the original dataset has been created as 'diabetes_unclean_backup.csv'.")

st.subheader("Check for Null values")
st.write("Checking for null values in the dataset:")
col1, col2 = st.columns(2)

with col1:
    st.caption("Null Values Count")
    null_values = dataset.isnull().sum()
    st.write(null_values[null_values > 0])

with col2:
    st.caption("Rows with Null Values")
    total_rows = dataset.shape[0]
    rows_with_nan = dataset.isnull().any(axis=1).sum()
    percentage_with_nan = (rows_with_nan / total_rows) * 100
    st.write(f"Total Rows: {total_rows}")
    st.write(f"Rows with Null Values: {rows_with_nan}")
    st.write(f"Percentage of Rows with Null Values: {percentage_with_nan:.2f}%")

    st.write(f"The rows with null values constitute only {percentage_with_nan:.2f}% of the dataset, which is less than 2%.")
    st.write("Hence, we can safely drop these rows without significantly impacting the dataset.")
    dataset = dataset.dropna()
    st.write("Rows with null values have been removed.")

# Conversion to numerical values
st.subheader("Conversion to Numerical Values")
st.markdown("""
In this section, we normalize and encode categorical columns to numerical values for further analysis and modeling.
""")

# Repairing inconsistent values and encoding of the 'Gender' column
st.subheader("Encoding 'Gender' Column")
col1, col2 = st.columns(2)

with col1:
    st.caption("Before Encoding")
    st.write("Unique values:", dataset['Gender'].unique())

with col2:
    st.caption("After Encoding")
    dataset['Gender'] = dataset['Gender'].str.upper().map({'M': 0, 'F': 1})
    st.write("Unique values:", dataset['Gender'].unique())

st.markdown("""Encoding applied: ```[0 = Male, 1 = Female]```""")
# Encoding of the 'CLASS' column
st.subheader("Encoding 'CLASS' Column")
col1, col2 = st.columns(2)

with col1:
    st.caption("Before Encoding")
    st.write("Unique values:", dataset['CLASS'].unique())

with col2:
    st.caption("After Encoding")
    dataset['CLASS'] = dataset['CLASS'].str.strip().str.upper().replace({'P': 'Y'}).map({'Y': 1, 'N': 0})
    st.write("Unique values:", dataset['CLASS'].unique())

st.markdown("""Encoding applied: ```[0 = Negative to diabetes, 1 = Positive to diabetes]```""")

# Page break
st.markdown("---")


# Unused columns
st.subheader("Removing Unused Columns")
st.markdown("""
Some columns are not relevant to predict diabetes. For example, columns like `ID` and `No_Pation` do not provide meaningful information for prediction and can be removed.
""")

# Backup of the cleaned and encoded dataset
ds_backup = dataset.copy()

# Removing unused columns
unused_columns = ['ID', 'No_Pation']
dataset.drop(columns=unused_columns, inplace=True)
st.write(f"The following columns have been removed: {unused_columns}")

# Count the occurrences of each class
class_counts = dataset['CLASS'].value_counts()

# Calculate the percentage
class_percentages = (class_counts / len(dataset)) * 100

# Page break
st.markdown("---")



# Outlier Detection and Handling Section
st.header("Outlier Detection and Handling")

# Statistical Summary
st.subheader("Statistical Summary of Features")
columns_to_drop = [col for col in ['Gender', 'AGE'] if col in dataset.columns]
stats = dataset.drop(columns=columns_to_drop).describe()
stats_to_show = stats.loc[['mean', 'std', 'min', 'max']]
st.write("Statistical summary of features:")
st.write(stats_to_show)
# Outlier Detection Analysis
st.markdown("# Outlier Detection Analysis")

st.markdown("""
## Features Requiring Special Attention

As shown in the preliminary analysis, the features **'Urea', 'Cr', 'TG', 'HDL',** and **'VLDL'** show high values of standard deviation ($\\sigma$). Specifically, these columns have standard deviation values higher than half their respective means:

$$\\text{Attention required when:}\\quad \\sigma > \\frac{\\text{mean}}{2}$$

This indicates potential outliers that warrant further investigation.

## Physiological Limits for Outlier Detection

Physiological ranges and limits:

| Feature | Lower Limit | Upper Limit | Justification |
|---------|-------------|-------------|---------------|
| **Urea** | 1 mmol/L | 25 mmol/L | Values outside this range are extremely rare in living patients and likely represent measurement errors |
| **Cr (Creatinine)** | 10 µmol/L | 400 µmol/L | Values above 400 µmol/L may indicate severe renal failure but could also be data entry errors; values below 10 µmol/L are biologically implausible |
| **TG (Triglycerides)** | 0.1 mmol/L | 10 mmol/L | While normal range is <1.7 mmol/L, values up to 10 mmol/L can occur in severe hypertriglyceridemia or **diabetic** patients |
| **HDL** | 0.3 mmol/L | 5 mmol/L | Values above 5 mmol/L are highly improbable and likely due to laboratory errors or data entry mistakes |
| **VLDL** | 0.05 mmol/L | ? mmol/L | Value derived from a medical equation that considers the levels of TG and cholesterol in the blood. Further investigation needed, possible synthetic values not needed for this project. |

*! Disclaimer*: The reference values provided are not intended for medical purposes and were obtained through online research. They are not guaranteed to be reliable or representative of the sample in question, as this is not the primary objective of the project.
Sources: https://www.scymed.com, https://www.my-personaltrainer.it/salute/conversione-colesterolo.html
                  
""")

# Identify features with high standard deviation
features_high_std = stats.columns[stats.loc['std'] > (stats.loc['mean'] / 2)].tolist()
st.write("Features with high standard deviation:", features_high_std)

# Physiological limits for outlier detection
st.subheader("Physiological limits vectors")
temp_limits_upper = [25, 400, 10, 5, 40]
temp_limits_lower = [1, 10, 0.1, 0.3, 0.05]

limits_df = pd.DataFrame({
    "Feature": features_high_std,
    "Lower Limit": temp_limits_lower,
    "Upper Limit": temp_limits_upper
})

st.write("Physiological limits for features with high standard deviation:")
st.dataframe(limits_df)

# Boxplots for suspicious features
st.subheader("Boxplots for Suspicious Features")
for feature, lower, upper in zip(features_high_std, temp_limits_lower, temp_limits_upper):
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.boxplot(data=dataset[feature], orient='h', ax=ax)
    ax.axvline(x=lower, color='green', linestyle='--', label='Lower Threshold')
    ax.axvline(x=upper, color='red', linestyle='--', label='Upper Threshold')
    ax.set_title(f'Boxplot of {feature}')
    ax.legend()
    st.pyplot(fig)

# Create a DataFrame to store the results
threshold_table = pd.DataFrame({
    'Feature': features_high_std,    # List of features with high standard deviation
    'Upper Threshold': temp_limits_upper,  # Corresponding upper thresholds for each feature
    'Lower Threshold': temp_limits_lower,  # Corresponding lower thresholds for each feature
    'Above Upper Threshold': [
        dataset[feature][dataset[feature] > temp_limits_upper[i]].count()  # Count values above the upper threshold
        for i, feature in enumerate(features_high_std)
    ],
    'Below Lower Threshold': [
        dataset[feature][dataset[feature] < temp_limits_lower[i]].count()  # Count values below the lower threshold
        for i, feature in enumerate(features_high_std)
    ]
})
# Add columns to the threshold table for the percentage of outliers
threshold_table['% Above Upper Threshold'] = (threshold_table['Above Upper Threshold'] / len(dataset)) * 100
threshold_table['% Below Lower Threshold'] = (threshold_table['Below Lower Threshold'] / len(dataset)) * 100

# Display the table
st.subheader("Threshold Analysis Table")
st.write(threshold_table)


# Replace outliers with NaN
st.subheader("Replacing Outliers with NaN")
st.markdown("""
Outliers detected in the dataset are replaced with NaN values. This approach allows us to handle extreme values without immediately removing them, preserving the dataset's integrity for further analysis. If the proportion of NaN values becomes significant, additional measures will be considered.
""")

for i, feature in enumerate(features_high_std):
    lower_limit = temp_limits_lower[i]
    upper_limit = temp_limits_upper[i]
    values_to_replace = (dataset[feature] < lower_limit) | (dataset[feature] > upper_limit)
    dataset.loc[values_to_replace, feature] = np.nan

# Outlier Removal
for i, feature in enumerate(features_high_std):
    lower_limit = temp_limits_lower[i]
    upper_limit = temp_limits_upper[i]
    values_to_remove = (dataset[feature] < lower_limit) | (dataset[feature] > upper_limit)
    dataset.loc[values_to_remove, feature] = None

st.markdown("""---""")



# Scatterplot for TG vs VLDL
st.subheader("Scatterplot: TG vs VLDL")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=dataset, x='TG', y='VLDL', ax=ax, alpha=0.5)
ax.set_title("Scatterplot: TG vs VLDL")
st.pyplot(fig)

# VLDL and TG Analysis
st.markdown("### VLDL and TG")
st.markdown("""
Since VLDL is a value typically synthetic, we investigate its distribution in relation to TG. The scatter plot below highlights their relationship, showing a positive correlation between the two variables.

VLDL can also be measured directly in some cases, thus the relation with the TG value can be non-linear in some cases.

Formulas for computing VLDL starting from TG:
            
$$VLDL = \\frac{\\text{TG}}{5} \\quad \\text{(mg/dL)}$$ 
            
$$VLDL = \\frac{\\text{TG}}{2.2} \\quad \\text{(mmol/L)}$$

#### Conversion rate between different units of measurement:
##### (TG): 
$$\\text{mmol/L} = \\frac{\\text{mg/dL}}{88.5}$$
$$\\text{mg/dL} = \\text{mmol/L} \\times 88.5$$

##### Cholesterol (LDL, HDL, VLDL, Total): 
$$\\text{mmol/L} = \\frac{\\text{mg/dL}}{38.67}$$
$$\\text{mg/dL} = \\text{mmol/L} \\times 38.67$$
""")

# Convert VLDL values greater than 4
dataset.loc[dataset['VLDL'] > 4, 'VLDL'] = (dataset['VLDL'] * 5.5 / 38.67) / 2.2

# Scatterplot after conversion
st.subheader("Scatterplot After VLDL Conversion")
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=dataset, x='TG', y='VLDL', ax=ax, alpha=0.5)
ax.set_title("Scatterplot: TG vs VLDL (After Conversion)")
st.pyplot(fig)

# Handle missing values
st.subheader("Handling Missing Values")
rows_with_nan = dataset.isnull().any(axis=1).sum()
percentage_with_nan = (rows_with_nan / len(dataset)) * 100
st.write(f"Number of rows with NaN values: {rows_with_nan}")
st.write(f"Percentage of rows with NaN values: {percentage_with_nan:.2f}%")

dataset.dropna(inplace=True)
st.write("Rows with NaN values have been removed.")
st.write("Updated dataset shape:", dataset.shape)


















# EDA Section
st.header("Data Exploration")
st.header("Categorical Features and population overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Gender Distribution:")
    gender_counts = dataset['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax)
    ax.set_xticklabels(['Male', 'Female'])
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.write("The population is approximately evenly distributed between males and females, with a slight majority of males.")

with col2:
    st.subheader("Patients with Diabetes:")
    class_counts = dataset['CLASS'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_xticklabels(['No Diabetes (0)', 'Has Diabetes (1)'])
    ax.set_xlabel("Diabetes Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.write("The population is imbalanced, with the majority of patients having diabetes. This imbalance should be taken into account when making predictions about the likelihood of diabetes in patients.")

with col3:
    st.subheader("Age Distribution (KDE)")
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(dataset['AGE'], shade=True, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    st.pyplot(fig)
    st.write("The population's age distribution KDE plot shows a peak in the range of 50-60, and it is approximately bell-shaped, indicating a normal-like distribution.")

st.markdown("---")


# Feature Distribution Section
st.header("Feature Distribution")
selected_feature = st.selectbox("Select a feature to visualize its distribution", dataset.columns)

fig, ax = plt.subplots(figsize=(7, 5))
sns.kdeplot(dataset[selected_feature], shade=True, ax=ax)
ax.set_xlabel(selected_feature)
ax.set_ylabel("Density")
st.pyplot(fig)

# Scatterplot Section
st.header("Scatterplot Visualization")
st.markdown("Select two features to visualize their relationship using a scatterplot.")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.selectbox("Select Feature 1", dataset.columns)

with col2:
    feature2 = st.selectbox("Select Feature 2", dataset.columns)

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=dataset[feature1], y=dataset[feature2], ax=ax)
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
st.pyplot(fig)


# Correlation Heatmap
st.header("Correlation heatmap")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'shrink': 0.8}, linewidths=0.5, ax=ax)
plt.xticks(rotation=90, ha='right')
plt.yticks()
st.pyplot(fig)






# Prediction Section
st.subheader("Prediction Models")
target_column = st.selectbox("Select Target Column", dataset.columns)

if st.button("Train Model"):
    # Splitting data
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(plt)
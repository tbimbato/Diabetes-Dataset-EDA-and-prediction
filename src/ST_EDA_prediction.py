# Imports:

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.patches import Patch

# ======================================================================
# to run streamlit webapp run: 'streamlit run ST_EDA_prediciton.py' in the terminal (same dir of python script)
# ======================================================================

def user_input() -> pd.DataFrame:
    """
    Create a form for user input of patient data.
    Returns:
        pd.DataFrame: DataFrame containing the user input values.
    """

    # Create a temporary dictionary to store input data:
    input_data = {}
    
    st.markdown("### Enter Patient Information")
    
    # Create multiple columns for better layout
    col1, col2, col3 = st.columns(3)
    

    with col1:
        input_data['Gender'] = st.selectbox('Gender', options=['Male', 'Female'], index=0)
        input_data['AGE'] = st.slider('Age', min_value=20, max_value=90, value=29, step=1)
        input_data['Urea'] = st.slider('Urea (mmol/L)', min_value=1.0, max_value=25.0, value=4.5, step=0.1)
        input_data['Cr'] = st.slider('Creatinine (μmol/L)', min_value=10.0, max_value=400.0, value=90.0, step=1.0)
    
    with col2:
        input_data['HbA1c'] = st.slider('HbA1c (%)', min_value=3.0, max_value=15.0, value=5.2, step=0.1)
        input_data['Chol'] = st.slider('Cholesterol (mmol/L)', min_value=2.0, max_value=10.0, value=4.5, step=0.1)
        input_data['TG'] = st.slider('Triglycerides (mmol/L)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        input_data['HDL'] = st.slider('HDL (mmol/L)', min_value=0.3, max_value=5.0, value=1.3, step=0.1)
    
    with col3:
        input_data['LDL'] = st.slider('LDL (mmol/L)', min_value=0.5, max_value=7.0, value=2.7, step=0.1)
        input_data['VLDL'] = st.slider('VLDL (mmol/L)', min_value=0.05, max_value=3.0, value=0.45, step=0.05)
        input_data['BMI'] = st.slider('BMI (kg/m²)', min_value=15.0, max_value=45.0, value=22.0, step=0.1)

    # Convert gender data to numerical values
    if input_data['Gender'] == 'Male':
        input_data['Gender'] = 0
    else:  # 'Female'
        input_data['Gender'] = 1
    
    # Convert the input dictionary to a dataframe
    input_df = pd.DataFrame([input_data])
    
    st.subheader("Summary of entered patient Data")
    st.write(input_df)

    # return the dataframe with all the input data
    return input_df



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


# Load default dataset ('diabetes_unclean.csv')
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

st.subheader("Backup of the original dataset")
ds_backup = dataset.copy()
ds_backup.to_csv('../datasets/diabetes_unclean_backup.csv', index=False)
st.write("Backup of the original dataset has been created as 'diabetes_unclean_backup.csv'.")

st.subheader("Null values")
st.write("Checking for null values in the dataset:")
col1, col2 = st.columns(2) # 2 columns layout

# column 1:
with col1: 
    st.caption("Null Values Count")
    null_values = dataset.isnull().sum()
    st.write(null_values[null_values > 0])

# column 2:
with col2:
    st.caption("Rows with Null Values")
    total_rows = dataset.shape[0]
    rows_with_nan = dataset.isnull().any(axis=1).sum()
    percentage_with_nan = (rows_with_nan / total_rows) * 100
    st.write(f"Total Rows: {total_rows}")
    st.write(f"Rows with Null Values: {rows_with_nan}")
    st.write(f"Percentage of Rows with Null Values: {percentage_with_nan:.2f}%")

st.write(f"The rows with null values constitute only {percentage_with_nan:.2f}% of the dataset, which is less than 2%. Hence, we can safely drop these rows without significantly impacting the dataset.")
dataset = dataset.dropna()
st.write("Rows with null values have been removed.")

# Conversion to numerical values
st.subheader("Conversion to Numerical Values")
st.markdown("""
In this section, we normalize and encode categorical columns to numerical 
            values for further analysis and modeling.
""")

# Repairing inconsistent values and encoding of the 'Gender' column
st.subheader("Encoding 'Gender' Column")
col1, col2 = st.columns(2)

with col1:
    st.caption("Gender - Before Encoding")
    st.write("Unique values:", dataset['Gender'].unique())

with col2:
    st.caption("Gender - After Encoding")
    dataset['Gender'] = dataset['Gender'].str.upper().map({'M': 0, 'F': 1})
    st.write("Unique values:", dataset['Gender'].unique())

st.markdown("""
**Gender Encoding:**
- 0 = Male
- 1 = Female
""")

# Encoding of the 'CLASS' column
st.subheader("Encoding 'CLASS' Column")
col1, col2 = st.columns(2)

with col1:
    st.caption("CLASS - Before Encoding")
    st.write("Unique values:", dataset['CLASS'].unique())

with col2:
    st.caption("CLASS - After Encoding")
    dataset['CLASS'] = dataset['CLASS'].str.strip().str.upper().replace({'P': 'Y'}).map({'Y': 1, 'N': 0})
    st.write("Unique values:", dataset['CLASS'].unique())

st.markdown("""
**CLASS Encoding:**
- 0 = Negative (No Diabetes)
- 1 = Positive (Has Diabetes)
""")

# Page break
st.markdown("---")


# Unused columns
st.subheader("Removing Unused Columns")
st.markdown("""
Some columns are not relevant to predict diabetes. For example,
            columns like `ID` and `No_Pation` do not provide
            meaningful information for prediction and can be removed.
""")

# Backup of the cleaned and encoded dataset
ds_backup = dataset.copy()

# Removing unused columns
unused_columns = ['ID', 'No_Pation']
dataset.drop(columns=unused_columns, inplace=True)
st.write(f"The following columns have been removed: {unused_columns}")

# Count the occurrences of each class
class_counts = dataset['CLASS'].value_counts()
# Page break
st.markdown("---")



# Outlier Detection and Handling Section
st.header("Outlier Detection and Handling")

# Statistical Summary
st.subheader("Statistical Summary of Features")
columns_to_drop = ['Gender', 'AGE']
stats = dataset.drop(columns=columns_to_drop).describe() # computing stats excluding 'Gender' and 'AGE'
stats_to_show = stats.loc[['mean', 'std', 'min', 'max']] # preparing to show only mean, std, min and max

st.write("Statistical summary of features:")
st.write(stats_to_show)

# ======================================================================
# Outlier Detection Analysis
st.subheader("Outlier Detection Analysis")
st.markdown("""
#### Features Requiring Special Attention

As shown in the preliminary analysis, the features **'Urea', 'Cr', 'TG', 
            'HDL',** and **'VLDL'** show high values of standard deviation ($\\sigma$).
            Specifically, these columns have standard deviation values higher than half their respective means:

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
# features with high standard deviation <- feature where 'std' > ('mean'/2) (we use 'loc(ate)' to get the index of the row)
features_high_std = stats.columns[stats.loc['std'] > (stats.loc['mean'] / 2)].tolist()
st.write("Features with high standard deviation:", features_high_std)

# Physiological limits for outlier detection
st.subheader("Physiological limits vectors")

# LIMITS DEFINED -- SEE LINK TO REFERENCE
temp_limits_upper = [25, 400, 12, 6, 40]
temp_limits_lower = [1, 10, 0.1, 0.3, 0.05]

# suspicious features were identified in the notebook,
# Here we reproducing the process, assuming to find the same features
# And we apply the limits to the same features


# projection to dataframe
limits_df = pd.DataFrame({
    "Feature": features_high_std,     # vector of features with high standard deviation
    "Lower Limit": temp_limits_lower, # vector of lower limits
    "Upper Limit": temp_limits_upper  # vector of upper limits
})

st.write("Physiological limits for features with high standard deviation:")
st.dataframe(limits_df)

# Boxplots for suspicious features
st.subheader("Boxplots for Suspicious Features")
st.markdown("Visualizing the distribution of features with high standard deviation to identify outliers.")

# for each feature with high standard deviation, create a boxplot with upper and lower limits
for i, (feature, lower, upper) in enumerate(zip(features_high_std, temp_limits_lower, temp_limits_upper)):
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Create horizontal boxplot
    sns.boxplot(data=dataset[feature], orient='h', ax=ax, color='lightblue')
    
    # Add threshold lines with better visibility
    ax.axvline(x=lower, color='blue', linestyle='--', linewidth=1.5, label=f'Lower Threshold ({lower})') # write correct threshold in the current cycle (for..)
    ax.axvline(x=upper, color='red', linestyle='--', linewidth=1.5, label=f'Upper Threshold ({upper})')
    
    # Title, labels, legend, grid
    ax.set_title(f'Distribution of {feature}', fontsize=12)
    ax.set_xlabel(f'{feature} Value', fontsize=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    st.markdown("---")


# Create a DataFrame to store the results
threshold_table = pd.DataFrame({
    'Feature': features_high_std,           # List of features with high standard deviation
    'Upper Threshold': temp_limits_upper,   # Corresponding upper thresholds for each feature
    'Lower Threshold': temp_limits_lower,   # Corresponding lower thresholds for each feature
    'Above Upper Threshold': [
        dataset[feature][dataset[feature] > temp_limits_upper[i]].count()   # Count values above upper threshold
        for i, feature in enumerate(features_high_std)
    ],
    'Below Lower Threshold': [
        dataset[feature][dataset[feature] < temp_limits_lower[i]].count()   # Count values below lower threshold
        for i, feature in enumerate(features_high_std)
    ]
})

# Storing the percentage of outliers
threshold_table['% Above Upper Threshold'] = (threshold_table['Above Upper Threshold'] / len(dataset)) * 100
threshold_table['% Below Lower Threshold'] = (threshold_table['Below Lower Threshold'] / len(dataset)) * 100

# Display the table
st.subheader("Threshold Analysis Table")
st.dataframe(threshold_table)

# Replace outliers with NaN
st.subheader("Replacing Outliers with NaN")
st.markdown("""
Outliers detected in the dataset are replaced with NaN values. 
            This approach allows us to handle extreme values without immediately removing them, preserving the dataset's integrity for further analysis. 
            If the proportion of NaN values becomes significant, additional measures will be considered.
""")

# Handle outliers for features with high standard deviation
for i, feature in enumerate(features_high_std):
    lower_limit = temp_limits_lower[i]
    upper_limit = temp_limits_upper[i]
    # Create a mask for outliers where values are outside the defined limits
    outlier_mask = (dataset[feature] < lower_limit) | (dataset[feature] > upper_limit)

    # Replace outliers with NaN
    dataset.loc[outlier_mask, feature] = np.nan

st.markdown("---")

# Scatterplot for TG vs VLDL
st.subheader("TG and VLDL Analysis")
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(data=dataset, x='TG', y='VLDL', ax=ax, alpha=0.3)
ax.set_title("Scatterplot: TG vs VLDL")
# Highlight points where VLDL > 4
highlight = dataset['VLDL'] > 4
ax.scatter(dataset.loc[highlight, 'TG'], dataset.loc[highlight, 'VLDL'], 
           color='red', label='VLDL > 5', edgecolor='black')
ax.legend()
st.pyplot(fig)

# VLDL and TG Analysis
st.markdown("### VLDL and TG")
st.markdown("""
Since VLDL is a value typically synthetic, we investigate its distribution in relation to TG. 
            The scatter plot below highlights their relationship, showing a positive correlation between the two variables.

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
# see notebook for reference
dataset.loc[dataset['VLDL'] > 4, 'VLDL'] = (dataset['VLDL'] * 5.5 / 38.67) / 2.2

# Scatterplot after conversion
st.subheader("Scatterplot After VLDL Conversion")
fig, ax = plt.subplots(figsize=(12, 8))
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


# ======================================================================
# EDA Section
st.header("Data Exploration")
st.header("Categorical Features and population overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Gender Distribution:")
    gender_counts = dataset['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax)
    ax.set_xticklabels(['Male', 'Female'])
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.write("The population is approximately evenly distributed between males and females, with a slight majority of males.")

with col2:
    st.subheader("Patients with Diabetes:")
    class_counts = dataset['CLASS'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
    ax.set_xticklabels(['No Diabetes (0)', 'Has Diabetes (1)'])
    ax.set_xlabel("Diabetes Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.write("The population is imbalanced, with the majority of patients having diabetes. " \
    "This imbalance should be taken into account when making predictions about the likelihood of diabetes in patients.")

with col3:
    st.subheader("Age Distribution (KDE)")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.kdeplot(dataset['AGE'], shade=True, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    st.pyplot(fig)
    st.write("The population's age distribution KDE plot shows a peak in the range of 50-60, "
    "and it is approximately bell-shaped, indicating a normal-like distribution.")

st.markdown("---")


# Feature Distribution Section
st.header("Feature Distribution")
selected_feature = st.selectbox("Select a feature to visualize its distribution", dataset.columns)

fig, ax = plt.subplots(figsize=(12, 8))
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

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x=dataset[feature1], y=dataset[feature2], ax=ax)
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
st.pyplot(fig)


# Correlation Heatmap
st.header("Correlation heatmap")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'shrink': 0.8}, linewidths=0.5, ax=ax)
plt.xticks(rotation=90, ha='right')
plt.yticks()
st.pyplot(fig)

# Correlation with Target Analysis
st.markdown("---")
st.header("Features Most Correlated with Diabetes")
st.markdown("""
We can see in the correlation heatmap that the features more correlated with CLASS (diabetes status) are:
- **HbA1c**: Hemoglobin A1c percentage (indicates average blood sugar levels)
- **BMI**: Body Mass Index
- **AGE**: Patient's age
    
Let's explore these key features in more detail.
""")

# Create plots for these key features
key_features = ['HbA1c', 'BMI', 'AGE']

# Box plots comparing distribution by diabetes status (horizontal layout)
st.subheader("Distribution Comparison by Diabetes Status")

# Create a single figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(12, 8))

# Create boxplots for each key feature
sns.boxplot(x='CLASS', y='HbA1c', data=dataset, ax=axes[0], palette=['lightblue', 'salmon'])
sns.boxplot(x='CLASS', y='BMI', data=dataset, ax=axes[1], palette=['lightblue', 'salmon'])
sns.boxplot(x='CLASS', y='AGE', data=dataset, ax=axes[2], palette=['lightblue', 'salmon'])

# Set titles and labels
axes[0].set_title("HbA1c by Diabetes Status")
axes[1].set_title("BMI by Diabetes Status")
axes[2].set_title("Age by Diabetes Status")

# Set x-axis labels
for ax in axes:
    ax.set_xlabel("Diabetes Status (0=No, 1=Yes)")

# Adjust layout to make room for text
plt.tight_layout()
st.pyplot(fig)

# Add interpretations under each plot
st.markdown("""
**Key Observations**:
- **HbA1c**: Clearly higher values in diabetic patients, as expected since it directly measures blood glucose control.
- **BMI**: Diabetic patients tend to have higher BMI values, though significant overlap exists on the lower end.
- **Age**: Diabetic patients are generally older, but the wide distribution shows diabetes affects people across varied age ranges.
""")

# 3D scatter to visualize the three features
st.subheader("3D Visualization of Key Features")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i, label in enumerate(['Non-Diabetic', 'Diabetic']):
    subset = dataset[dataset['CLASS'] == i]
    ax.scatter(
        subset['HbA1c'], 
        subset['BMI'], 
        subset['AGE'],
        label=label,
        alpha=0.6,
        s=25
    )
ax.set_xlabel('HbA1c')
ax.set_ylabel('BMI')
ax.set_zlabel('Age', labelpad=-8)
ax.legend()
ax.set_title('3D Scatter Plot of Key Diabetes Predictors')
st.pyplot(fig)
st.markdown("""
        **Observation**: Despite the strong imbalance in the dataset, we can observe a distinct clustering of 
            patients with similar characteristics who don't have diabetes (represented by blue points). T
            hese non-diabetic patients generally show lower values across all three key predictors: HbA1c, BMI, and age. 
            While there is some overlap between the two groups, particularly in the middle ranges, the separation 
            is still visible, confirming that these three features together provide meaningful predictive power for diabetes status. 
            The diabetic patients points tend to occupy the higher ranges of these measurements, especially for HbA1c values.
        """)



st.markdown("""---""")

# Prediction Section
st.header("Model Selection and Training")

# Define target and features
X = dataset.drop(columns=['CLASS'])
y = dataset['CLASS']

# Train-test splitting
st.markdown("""
### Train-Test Split
Training a machine learning model requires dividing the dataset into:
- **Training set**: Used to train the model (typically 70-80% of data)
- **Testing set**: Used to evaluate model performance on unseen data

""")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=95)

# Class imbalance
st.markdown("""
### Class Imbalance in Diabetes Dataset
This dataset has more diabetic than non-diabetic patients, creating a class imbalance.
Training on imbalanced data can lead to:
- Models biased toward the majority class
- Poor predictive performance on the minority class
- Misleadingly high accuracy metrics
""")

# Create a dataframe with features and target for the training set
train_df = pd.concat([X_train, y_train], axis=1)

# Split by class
class_0 = train_df[train_df['CLASS'] == 0]
class_1 = train_df[train_df['CLASS'] == 1]

# Display class distribution before balancing
st.subheader("Class Distribution in Training Set")
before_counts = y_train.value_counts()
fig, ax = plt.subplots(figsize=(12, 4))
sns.barplot(x=before_counts.index, y=before_counts.values, ax=ax)
ax.set_xticklabels(['No Diabetes (0)', 'Has Diabetes (1)'])
ax.set_ylabel("Count")
st.pyplot(fig)
st.write(f"Class 0 (No Diabetes): {len(class_0)} samples")
st.write(f"Class 1 (Has Diabetes): {len(class_1)} samples")
st.write(f"Ratio of diabetic to non-diabetic samples: {len(class_1)/len(class_0):.2f}")

# Create balanced dataset
st.markdown("""
### Balancing the Dataset
We'll use **random undersampling** of the majority class to create a balanced dataset:
- Take all samples from the minority class (non-diabetic)
- Randomly select an equal number of samples from the majority class (diabetic)
- Combine them to create a balanced training set

This approach prevents the model from being biased toward predicting diabetes simply because it's more common in the dataset.
""")

# Undersample the majority class
class_1_under = class_1.sample(n=len(class_0), random_state=95)
train_balanced = pd.concat([class_0, class_1_under]).sample(frac=1, random_state=95).reset_index(drop=True)
X_train_bal = train_balanced.drop(columns=['CLASS'])
y_train_bal = train_balanced['CLASS']

# Plot the effect of balancing on class distribution
after_counts = y_train_bal.value_counts()
fig, ax = plt.subplots(figsize=(12, 4))
sns.barplot(x=after_counts.index, y=after_counts.values, ax=ax)
ax.set_xticklabels(['No Diabetes (0)', 'Has Diabetes (1)'])
ax.set_ylabel("Count")
ax.set_title("Class Distribution After Undersampling")
st.pyplot(fig)
st.write(f"After balancing: Class 0 (No Diabetes): {after_counts[0]} samples, Class 1 (Has Diabetes): {after_counts[1]} samples")

st.subheader("Models Used for Prediction")


# ======================================================================
# Models and predictions
# (random_state = 95 is set for reproducibility of results.)

# imbalanced:
log_reg =       LogisticRegression(random_state=95) 
decision_tree = DecisionTreeClassifier(random_state=95)
knn =           KNeighborsClassifier()
rnd_forest =    RandomForestClassifier(random_state=95)

# balanced:
bal_log_reg =       LogisticRegression(random_state=95) 
bal_decision_tree = DecisionTreeClassifier(random_state=95)
bal_knn =           KNeighborsClassifier()
bal_rnd_forest =    RandomForestClassifier(random_state=95)

# Display available machine learning models
st.markdown("""
We will use the following machine learning models:
- **Logistic Regression**: A linear model for binary classification
- **Decision Tree**: A tree-based model that makes decisions based on feature thresholds
- **K-Nearest Neighbors**: A non-parametric model that classifies based on closest training examples
- **Random Forest**: An ensemble model that combines multiple decision trees for improved performance
""")

# ==== TRAINING AND TESTING ====

# Training for imbalanced datasets:
log_reg.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
knn.fit(X_train, y_train)
rnd_forest.fit(X_train, y_train)

# Training for BALANCED datasets:
bal_log_reg.fit(X_train_bal, y_train_bal)
bal_decision_tree.fit(X_train_bal, y_train_bal)
bal_knn.fit(X_train_bal, y_train_bal)
bal_rnd_forest.fit(X_train_bal, y_train_bal)

# Testing for imbalanced datasets:
prediction_log_reg = log_reg.predict(X_test)
prediction_decision_tree = decision_tree.predict(X_test)
prediction_knn = knn.predict(X_test)
prediction_r_forest = rnd_forest.predict(X_test)

# Testing for BALANCED datasets:
prediction_balanced_log_reg = bal_log_reg.predict(X_test)
prediction_balanced_decision_tree = bal_decision_tree.predict(X_test)
prediction_balanced_knn = bal_knn.predict(X_test)
prediction_balanced_r_forest = bal_rnd_forest.predict(X_test)



# ==== BENCHMARK ====

# Define the models' list
models = ['Logistic Regression',
          'Decision Tree',
          'K-Nearest Neighbors',
          'Random Forest'
          ]

# predictions for balanced datasets:
vector_prediction_balanced =    [prediction_balanced_log_reg,
                                 prediction_balanced_decision_tree,
                                 prediction_balanced_knn,
                                 prediction_balanced_r_forest
                                ]

# predictions for imbalanced datasets:
vector_prediction_imbalanced = [prediction_log_reg,
                                prediction_decision_tree,
                                prediction_knn,
                                prediction_r_forest
                                ]



# == BALANCED DATASETS ==
# preparing empty lists to store the metrics
balanced_accuracy =     []
balanced_precision =    []
balanced_f1 =           []
balanced_recall_0 =     []
balanced_recall_1 =     []

for balanced_prediction in vector_prediction_balanced:
    balanced_accuracy.append(accuracy_score(y_test, balanced_prediction))
    balanced_precision.append(precision_score(y_test, balanced_prediction))
    balanced_f1.append(f1_score(y_test, balanced_prediction))
    balanced_recall_0.append(recall_score(y_test, balanced_prediction, pos_label=0))
    balanced_recall_1.append(recall_score(y_test, balanced_prediction, pos_label=1))


# Create dataframe for balanced metrics
balanced_metrics_df = pd.DataFrame({
    'Model':        models,
    'Accuracy':     balanced_accuracy,
    'Precision':    balanced_precision,
    'F1-Score':     balanced_f1,
    'Recall (0)':   balanced_recall_0,
    'Recall (1)':   balanced_recall_1
})



# == imbalanced DATASETS ==
imbalanced_accuracy =   []
imbalanced_precision =  []
imbalanced_f1 =         []
imbalanced_recall_0 =   []
imbalanced_recall_1 =   []

for imbalanced_prediction in vector_prediction_imbalanced:
    imbalanced_accuracy.append(accuracy_score(y_test, imbalanced_prediction))
    imbalanced_precision.append(precision_score(y_test, imbalanced_prediction))
    imbalanced_f1.append(f1_score(y_test, imbalanced_prediction))
    imbalanced_recall_0.append(recall_score(y_test, imbalanced_prediction, pos_label=0))
    imbalanced_recall_1.append(recall_score(y_test, imbalanced_prediction, pos_label=1))

# Create dataframe for imbalanced metrics
imbalanced_metrics_df = pd.DataFrame({
    'Model':        models,
    'Accuracy':     imbalanced_accuracy,
    'Precision':    imbalanced_precision,
    'F1-Score':     imbalanced_f1,
    'Recall (0)':   imbalanced_recall_0,
    'Recall (1)':   imbalanced_recall_1
})

# ==== PLOTTING BENCHMARKS ====

# Benchmark all models
st.subheader("Performance Metrics Comparison")
st.markdown("Comparing model performance on balanced vs. imbalanced datasets")

# Prepare data for plotting
metrics = ['Accuracy', 'Precision', 'F1-Score', 'Recall (0)']

# ! pd.melt() remark: melt more columns into rows 
plot_data = pd.melt( 
    pd.concat([imbalanced_metrics_df.assign(Dataset='imbalanced'), balanced_metrics_df.assign(Dataset='Balanced')]),
    id_vars=['Model', 'Dataset'],
    value_vars=metrics,
    var_name='Metric',
    value_name='Score'
)


# Display the metrics dataframes
st.subheader("Balanced Dataset Metrics")
st.dataframe(balanced_metrics_df.round(2))

st.subheader("Imbalanced Dataset Metrics")
st.dataframe(imbalanced_metrics_df.round(2))



# plotting
for metric in metrics:
    metric_data = plot_data[plot_data['Metric'] == metric]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the pointplot
    sns.pointplot(
        data=metric_data, 
        x = "Dataset", 
        y = "Score", 
        hue = "Model", 
        dodge = 0.2,                         # separate overlapping points over X
    )
    
    # Set title and labels and room on y axis
    ax.set_ylim(0, 1.2) # add y-axis limits
    ax.set_title(f"{metric} Comparison")
    ax.set_xlabel("Dataset Type")
    ax.set_ylabel("Score")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02)) # set y-axis ticks intervals
    
    ax.legend()
    
    
    # Dynamically set y-axis limits
    min_score = metric_data['Score'].min()
    max_score = metric_data['Score'].max()
    ax.set_ylim(min_score -0.05, max_score + 0.05)
    
    # point values
    for line in ax.lines:
        for i in range(len(line.get_xdata())):
            x = line.get_xdata()[i]
            y = line.get_ydata()[i]
            ax.annotate( f'{y:.2f}', (x, y+0.005), color = line.get_color()) #ax.annotate(what to write, (x,y + offset) position)

    st.pyplot(fig)



# explanation of the metrics
st.markdown("""
### Metrics Interpretation
- **Accuracy**: The proportion of correct predictions (both true positives and true negatives) among all predictions.
- **Precision**: Of all the positive predictions, what proportion were actually positive (true positives / (true positives + false positives)).
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
- **Recall (0)**: The model's ability to correctly identify patients without diabetes (true negatives / (true negatives + false positives)).
""")

st.markdown("""
### Metrics results
Conclusions on Model Comparison (Balanced vs Imbalanced Dataset)

From the comparison, we observe that training on a balanced dataset generally improves the model's ability to identify both classes fairly, especially the minority class in the imbalanced scenario.
- Recall for class 0 (non-diabetic patients) significantly improves when using the balanced dataset. For example, the K-Nearest Neighbors model goes from 0.40 (imbalanced) to 0.80 (balanced), highlighting a better sensitivity to underrepresented cases.
- On the other hand, some models slightly lose accuracy when trained on the balanced dataset. This is expected, as accuracy can be misleading in imbalanced datasets and does not reflect the model's real performance across classes.
- Random Forest and Decision Tree remain the top-performing models in both scenarios, showing robustness to class imbalance and high performance across all metrics.
- Logistic Regression benefits from balancing in terms of recall and F1-Score, suggesting improved generalization.
- In general, while imbalanced training can lead to high precision and overall accuracy, it tends to bias the model toward the majority class, which is mitigated by balancing.            
""")

st.markdown("---")

# User input section
st.subheader("Make Predictions with Selected Model")


# default dataset is the balanced one:
balance_option = "Balanced dataset"

# User selects model and balancing option
col1, col2 = st.columns(2)
with col1:
    balance_option = st.radio("Dataset balancing:", ["Balanced dataset", "Original imbalanced dataset"])
with col2:
    selected_model_name = st.selectbox("Select a classification model:", options=["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Random Forest"])

# Recreate the dictionary based on the updated balance_option
models_dict = {
    'Logistic Regression': bal_log_reg if balance_option == "Balanced dataset" else log_reg,
    'Decision Tree': bal_decision_tree if balance_option == "Balanced dataset" else decision_tree,
    'K-Nearest Neighbors': bal_knn if balance_option == "Balanced dataset" else knn,
    'Random Forest': bal_rnd_forest if balance_option == "Balanced dataset" else rnd_forest
}


model = models_dict[selected_model_name]

training_set = "balanced" if balance_option == "Balanced dataset" else "imbalanced"

st.success(f"{selected_model_name} selected with {training_set} training data")

# !update: the models in models_dict are already trained, so we don't need to train them again

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
recall_1 = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred)

# Display metrics
st.subheader(f"Model Evaluation ({training_set} training)")

# metrics on two colimns
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("Accuracy", f"{acc:.2f}")
    st.metric("Precision", f"{prec:.2f}")
    st.metric("F1-Score", f"{f1:.2f}")

with metric_col2:
    st.metric("Recall (Non-Diabetic)", f"{recall_0:.2f}")
    st.metric("Recall (Diabetic)", f"{recall_1:.2f}")

# Feature importance (if Random Forest is selected, we can show it) 
if selected_model_name == "Random Forest":
    st.subheader("Feature Importances")
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette="viridis")
    ax.set_title(f"Feature Importance (Random Forest - {training_set} training)")
    st.pyplot(fig)
    
    st.markdown("""
    ### Feature Importance Interpretation
    The bars represent how much each feature contributed to the model's decisions.
    Longer bars indicate features that have greater influence on predicting diabetes status.
    """)


# Prediction on new input
st.header("Make a New Prediction")
st.markdown("Adjust the sliders below to input values and click 'Predict' to get a diabetes prediction.")


# Execute user_input() to create the input form
new_data = user_input() # calling the function to create the input form

# Add prediction button
if st.button("Predict"):
    # Check if model is trained, train it if not
    if not hasattr(model, "classes_"):  # This attribute exists after fitting
        st.warning("Model not trained yet. Training model with balanced dataset...")
        model.fit(X_train_bal, y_train_bal)
        st.success(f"{selected_model_name} trained on balanced dataset!")

    prediction = model.predict(new_data)[0]
    
    # Get prediction probability/confidence if the model supports it
    if hasattr(model, "predict_proba"):     #'hasattr' checks if the model has the method 'predict_proba' 
        proba = model.predict_proba(new_data)[0]
        confidence = proba[int(prediction)] * 100  # Convert to percentage
    else:
        confidence = "Not available for this model"
    

    # Create columns for better layout
    if prediction == 1:
        st.markdown("### Prediction:")
        st.markdown("**Diabetic** (Class 1)")
    else:
        st.markdown("### Prediction:")
        st.markdown("**Non-Diabetic** (Class 0)")

    if isinstance(confidence, float):
        st.markdown("### Confidence Level:")
        
        st.progress(confidence/100) #confidence as a st progress line
        st.markdown(f"**{confidence:.1f}%** confidence in this prediction") # <**> for bold MD type inside f-string
        
    else:
        st.markdown(f"**Confidence: {confidence}**")    


# Create a function to evaluate all models on user input (this function is not generic!)
def evaluate_all_models(input_data):
    # Define all models
    models_config = {
        'Logistic Regression (Balanced)': bal_log_reg,
        'Decision Tree (Balanced)': bal_decision_tree,
        'K-Nearest Neighbors (Balanced)': bal_knn,
        'Random Forest (Balanced)': bal_rnd_forest,
        'Logistic Regression (Imbalanced)': log_reg,
        'Decision Tree (Imbalanced)': decision_tree,
        'K-Nearest Neighbors (Imbalanced)': knn,
        'Random Forest (Imbalanced)': rnd_forest
    }
    
    # Store results
    results = []
    
    # Iterate through all models
    for model_name, model in models_config.items():
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get confidence if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            confidence = proba[int(prediction)] * 100
        else:
            confidence = None
            
        # Store results
        results.append({
            'Model': model_name,
            'Prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'Confidence': confidence,
            'Training': 'Balanced' if 'Balanced' in model_name else 'Imbalanced'
        })
    
    return pd.DataFrame(results)

# Button to run benchmark on all models
st.markdown("Run benchmark on all models to compare predictions and confidence levels on the same input data.")
if st.button("Benchmark All Models on This Input"):
    st.subheader("Comparing All Models on User Input and Displaying Results")
    
    # run func to evaluate all models
    benchmark_results = evaluate_all_models(new_data)
    
    # print result on the webapp
    st.dataframe(benchmark_results)

    st.subheader("Prediction Confidence by Model")
    
    benchmark_results['Color'] = benchmark_results['Prediction'].map({
        'Diabetic': 'red', 
        'Non-Diabetic': 'blue'
    })
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(benchmark_results['Model'], benchmark_results['Confidence'], color=benchmark_results['Color'])
    
    # Add confidence values to the end of each bar
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, # positioning
            f"{benchmark_results['Confidence'].iloc[i]:.1f}%",          # iloc is used to access the value in the DataFrame
            va='center'
        )
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Model Predictions with Confidence Levels')
    ax.set_xlim(0, 105)  # Set x-axis limit to accommodate text

    legend_handles = [                              # custom legend
        Patch(color='red', label='Diabetic'),       # red diabetic
        Patch(color='blue', label='Non-Diabetic')   # blue non-diabetic
    ]
    ax.legend(handles=legend_handles, title='Prediction')
    ax.grid(axis='x', linestyle='--', alpha=0.7)        # grid
    
    st.pyplot(fig)

# ======================================================================

# CONCLUSION
st.title("Conclusion")
st.markdown("""
### Summary of Findings
- The dataset was preprocessed to handle outliers and missing values.
- Exploratory Data Analysis (EDA) revealed key features correlated with diabetes status.
- Machine learning models were trained and evaluated on both balanced and imbalanced datasets.
- A comparison of model performance metrics highlighted the importance of balancing the dataset for better predictive accuracy.
            """)

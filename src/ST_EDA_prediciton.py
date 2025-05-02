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

# to run streamlit webapp run 'streamlit run ST_EDA_prediciton.py' in the terminal



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
col1, col2 = st.columns(2) # Create two columns for better layout

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
In this section, we normalize and encode categorical columns to numerical values for further analysis and modeling.
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
st.subheader("Outlier Detection Analysis")

st.markdown("""
#### Features Requiring Special Attention

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
st.markdown("Visualizing the distribution of features with high standard deviation to identify outliers.")

for i, (feature, lower, upper) in enumerate(zip(features_high_std, temp_limits_lower, temp_limits_upper)):
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create horizontal boxplot
    sns.boxplot(data=dataset[feature], orient='h', ax=ax, color='lightblue')
    
    # Add threshold lines with better visibility
    ax.axvline(x=lower, color='blue', linestyle='--', linewidth=1.5, label=f'Lower Threshold ({lower})')
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

# Add columns for the percentage of outliers
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

st.markdown("---")



# Scatterplot for TG vs VLDL
st.subheader("TG and VLDL Analysis")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=dataset, x='TG', y='VLDL', ax=ax, alpha=0.3)
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
fig, ax = plt.subplots(figsize=(10, 5))
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
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'salmon']
for i, label in enumerate(['Non-Diabetic', 'Diabetic']):
    subset = dataset[dataset['CLASS'] == i]
    ax.scatter(
        subset['HbA1c'], 
        subset['BMI'], 
        subset['AGE'],
        c=colors[i],
        label=label,
        alpha=0.6,
        s=50
    )
ax.set_xlabel('HbA1c')
ax.set_ylabel('BMI')
ax.set_zlabel('Age', labelpad=-8)
ax.legend()
ax.set_title('3D Scatter Plot of Key Diabetes Predictors')
st.pyplot(fig)
st.markdown("""
        **Observation**: Despite the strong imbalance in the dataset, we can observe a distinct clustering of patients with similar characteristics who don't have diabetes (represented by blue points). These non-diabetic patients generally show lower values across all three key predictors: HbA1c, BMI, and age. 
        While there is some overlap between the two groups, particularly in the middle ranges, the separation is still visible, confirming that these three features together provide meaningful predictive power for diabetes status. The salmon-colored points (diabetic patients) tend to occupy the higher ranges of these measurements, especially for HbA1c values.
        """)



st.markdown("""---""")

# Prediction Section
st.header("Model Selection and Training")

# Define target and features
target_feature = 'CLASS'
X = dataset.drop(columns=[target_feature])
y = dataset[target_feature]

# Train-test splitting
st.markdown("""
### Understanding Train-Test Split
Training a machine learning model requires dividing the dataset into:
- **Training set**: Used to train the model (typically 70-80% of data)
- **Testing set**: Used to evaluate model performance on unseen data

""")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

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
fig, ax = plt.subplots(figsize=(6, 4))
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
class_1_under = class_1.sample(n=len(class_0), random_state=42)
train_balanced = pd.concat([class_0, class_1_under]).sample(frac=1, random_state=42).reset_index(drop=True)
X_train_bal = train_balanced.drop(columns=['CLASS'])
y_train_bal = train_balanced['CLASS']

st.subheader("Models Used for Prediction")

# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=95),
    "Decision Tree": DecisionTreeClassifier(random_state=95),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=95)
}

# Display available machine learning models
st.markdown("""
We will use the following machine learning models:
- **Logistic Regression**: A linear model for binary classification
- **Decision Tree**: A tree-based model that makes decisions based on feature thresholds
- **K-Nearest Neighbors**: A non-parametric model that classifies based on closest training examples
- **Random Forest**: An ensemble model that combines multiple decision trees for improved performance
""")


# Benchmark all models
st.subheader("Model Benchmarks")
st.markdown("Comparing all models on both balanced and imbalanced datasets")

if st.button("Run All Benchmarks", type="primary"):
    with st.spinner("Running benchmarks on all models..."):
        # Prepare a dataframe to store results
        results = []
        
        # Run benchmarks for each model
        for model_name, model_obj in models.items():
            # Train on balanced dataset
            model_bal = models[model_name]
            model_bal.fit(X_train_bal, y_train_bal)
            y_pred_bal = model_bal.predict(X_test)
            
            # Calculate metrics for balanced training
            acc_bal = accuracy_score(y_test, y_pred_bal)
            prec_bal = precision_score(y_test, y_pred_bal)
            recall_bal = recall_score(y_test, y_pred_bal)
            f1_bal = f1_score(y_test, y_pred_bal)
            
            # Train on imbalanced dataset
            model_imbal = models[model_name]
            model_imbal.fit(X_train, y_train)
            y_pred_imbal = model_imbal.predict(X_test)
            
            # Calculate metrics for imbalanced training
            acc_imbal = accuracy_score(y_test, y_pred_imbal)
            prec_imbal = precision_score(y_test, y_pred_imbal)
            recall_imbal = recall_score(y_test, y_pred_imbal)
            f1_imbal = f1_score(y_test, y_pred_imbal)
            
            # Store results
            results.append({
                'Model': model_name,
                'Dataset': 'Balanced',
                'Accuracy': acc_bal,
                'Precision': prec_bal,
                'Recall': recall_bal,
                'F1': f1_bal
            })
            
            results.append({
                'Model': model_name,
                'Dataset': 'Imbalanced',
                'Accuracy': acc_imbal,
                'Precision': prec_imbal,
                'Recall': recall_imbal,
                'F1': f1_imbal
            })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Display results table
        st.write("Benchmark Results:")
        st.dataframe(results_df.style.format({
            'Accuracy': '{:.2f}',
            'Precision': '{:.2f}',
            'Recall': '{:.2f}',
            'F1': '{:.2f}'
        }))
        
        # Create visual comparison
        st.subheader("Visual Comparison of Models")
        
        # Reshape data for better visualization
        results_long = pd.melt(
            results_df, 
            id_vars=['Model', 'Dataset'], 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1'],
            var_name='Metric', 
            value_name='Score'
        )
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use seaborn for better visualization
        chart = sns.catplot(
            data=results_long,
            x='Model',
            y='Score',
            hue='Dataset',
            col='Metric',
            kind='bar',
            height=4,
            aspect=0.8,
            palette='viridis',
            alpha=0.8,
            legend_out=False
        )
        
        # Customize the plot
        chart.set_xticklabels(rotation=45, ha='right')
        chart.fig.suptitle('Model Performance Comparison', fontsize=16, y=1.05)
        chart.set_titles("{col_name}")
        
        # Display the plot
        st.pyplot(chart.fig)
        
        # Create heatmap for compact visualization
        st.subheader("Performance Heatmap")
        
        # Pivot the data for the heatmap
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
            heatmap_data = results_df.pivot(index='Model', columns='Dataset', values=metric)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu", 
                linewidths=0.5,
                vmin=0.5,
                vmax=1.0
            )
            plt.title(f"{metric} Comparison")
            st.pyplot(fig)


st.markdown("---")


# User selects model and balancing option
col1, col2 = st.columns(2)
with col1:
    selected_model_name = st.selectbox("Select a classification model:", list(models.keys()))
with col2:
    balance_option = st.radio("Dataset balancing:", ["Balanced dataset", "Original imbalanced dataset"])

model = models[selected_model_name] # we use the model selected by the user as key in the dictionary of models
# Train based on user selection
# if we want to use the balanced dataset:
if balance_option == "Balanced dataset":
    model.fit(X_train_bal, y_train_bal)
    st.success(f"{selected_model_name} trained on balanced dataset!")
    training_set = "balanced"



# if we want to use the unbalanced dataset:    
else:
    model.fit(X_train, y_train)
    st.success(f"{selected_model_name} trained on original imbalanced dataset!")
    training_set = "unbalanced"

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
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette="viridis")
    ax.set_title(f"Feature Importance (Random Forest - {training_set} training)")
    st.pyplot(fig)
    
    st.markdown("""
    ### Feature Importance Interpretation
    The bars represent how much each feature contributed to the model's decisions.
    Longer bars indicate features that have greater influence on predicting diabetes status.
    """)

# Explanation of metrics
st.markdown("""
### Understanding the Metrics
- **Accuracy**: Overall correctness of the model
- **Precision**: How many of the predicted diabetic cases are actually diabetic
- **Recall (Diabetic)**: How many actual diabetic cases the model correctly identified
- **Recall (Non-Diabetic)**: How many actual non-diabetic cases the model correctly identified
- **F1-Score**: Harmonic mean of precision and recall
""")


# Prediction on new input
st.header("Make a New Prediction")
st.markdown("Adjust the sliders below to input values and click 'Predict' to get a diabetes prediction.")

# Define a function to create the input form
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
        input_data['Gender'] = st.selectbox('Gender', options=['Male', 'Female'])
        input_data['AGE'] = st.slider('Age', min_value=20, max_value=90, value=50, step=1)
        input_data['Urea'] = st.slider('Urea (mmol/L)', min_value=1.0, max_value=25.0, value=5.0, step=0.1)
        input_data['Cr'] = st.slider('Creatinine (μmol/L)', min_value=10.0, max_value=400.0, value=90.0, step=1.0)
    
    with col2:
        input_data['HbA1c'] = st.slider('HbA1c (%)', min_value=3.0, max_value=15.0, value=5.7, step=0.1)
        input_data['Chol'] = st.slider('Cholesterol (mmol/L)', min_value=2.0, max_value=10.0, value=5.0, step=0.1)
        input_data['TG'] = st.slider('Triglycerides (mmol/L)', min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        input_data['HDL'] = st.slider('HDL (mmol/L)', min_value=0.3, max_value=5.0, value=1.2, step=0.1)
    
    with col3:
        input_data['LDL'] = st.slider('LDL (mmol/L)', min_value=0.5, max_value=7.0, value=3.0, step=0.1)
        input_data['VLDL'] = st.slider('VLDL (mmol/L)', min_value=0.05, max_value=3.0, value=0.6, step=0.05)
        input_data['BMI'] = st.slider('BMI (kg/m²)', min_value=15.0, max_value=45.0, value=25.0, step=0.1)
    

    # Convert geneder data to numerical values
    # Convert gender data to numerical values
    if input_data['Gender'] == 'Male':
        input_data['Gender'] = 0
    else:  # 'Female'
        input_data['Gender'] = 1
    
    # Convert the input dictionary to a dataframe
    input_df = pd.DataFrame([input_data])
    
    # Show the entered data
    st.subheader("Summary of entered patient Data")
    st.write(input_df)

    # return the dataframe with all the input data
    return input_df



# Execute the function to create the input form
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
    
    # Display the prediction result with class label and confidence
    st.subheader("Prediction Result:")
    
    # Create columns for better layout
    if prediction == 1:
        st.markdown("### Prediction:")
        st.markdown("**Diabetic** (Class 1)")
    else:
        st.markdown("### Prediction:")
        st.markdown("**Non-Diabetic** (Class 0)")

    if isinstance(confidence, float):
        st.markdown("### Confidence Level:")
        
        # Show confidence as percentage with progress bar
        st.progress(confidence/100)
        st.markdown(f"**{confidence:.1f}%** confidence in this prediction")
        
    else:
        st.markdown(f"**Confidence: {confidence}**")    

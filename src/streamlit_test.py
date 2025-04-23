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

#to run streamlit webapp run 'streamlit run streamlit_test.py' in the terminal

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
    fig, ax = plt.subplots(figsize=(10, 3))
    
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
threshold_table = pd.DataFrame({            # a dataframe containing the features with high std, upper and lower limits
    'Feature': features_high_std,           # List of features with high standard deviation
    'Upper Threshold': temp_limits_upper,   # Corresponding upper thresholds for each feature
    'Lower Threshold': temp_limits_lower,   # Corresponding lower thresholds for each feature
    'Above Upper Threshold': [
        #SELECT dataset[feature] WHERE dataset[feature] > upper limit for that feature
        dataset[feature][dataset[feature] > temp_limits_upper[i]].count()   # Count values above upper threshold for each feature in the list
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

st.markdown("---")



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
st.header("Model Selection and Training")

# Define target and features
target_feature = 'CLASS'
X = dataset.drop(columns=[target_feature])
y = dataset[target_feature]

# Educational box about train-test splitting
st.markdown("""
### Understanding Train-Test Split
Training a machine learning model requires dividing the dataset into:
- **Training set**: Used to train the model (typically 70-80% of data)
- **Testing set**: Used to evaluate model performance on unseen data

We use `stratify=y` to ensure both sets have the same class distribution proportion,
which is critical for imbalanced datasets like ours.
""")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Educational box about class imbalance
st.markdown("""
### Class Imbalance in Diabetes Dataset
This dataset has more diabetic than non-diabetic patients, creating a class imbalance.
Training on imbalanced data can lead to:
- Models biased toward the majority class
- Poor predictive performance on the minority class
- Misleadingly high accuracy metrics

To address this, we can either:
- Balance the dataset using undersampling/oversampling
- Use class weights in the model
- Apply specialized algorithms for imbalanced data
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

# Interactive model selection and training
st.subheader("Interactive Model Training")

# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=95),
    "Decision Tree": DecisionTreeClassifier(random_state=95),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=95)
}

# User selects model and balancing option
col1, col2 = st.columns(2)
with col1:
    selected_model_name = st.selectbox("Select a classification model:", list(models.keys()))
with col2:
    balance_option = st.radio("Dataset balancing:", ["Balanced dataset", "Original imbalanced dataset"])

model = models[selected_model_name]

# Train based on user selection
if st.button("Train Model", type="primary"):
    with st.spinner("Training model..."):
        if balance_option == "Balanced dataset":
            model.fit(X_train_bal, y_train_bal)
            st.success(f"{selected_model_name} trained on balanced dataset!")
            training_set = "balanced"
        else:
            model.fit(X_train, y_train)
            st.success(f"{selected_model_name} trained on original imbalanced dataset!")
            training_set = "imbalanced"
        
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
        
        # Create columns for metrics display
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("Accuracy", f"{acc:.2f}")
            st.metric("Precision", f"{prec:.2f}")
            st.metric("F1-Score", f"{f1:.2f}")
        
        with metric_col2:
            st.metric("Recall (Non-Diabetic)", f"{recall_0:.2f}")
            st.metric("Recall (Diabetic)", f"{recall_1:.2f}")
        
        # Explanation of metrics
        st.markdown("""
        ### Understanding the Metrics
        - **Accuracy**: Overall correctness of the model
        - **Precision**: How many of the predicted diabetic cases are actually diabetic
        - **Recall (Diabetic)**: How many actual diabetic cases the model correctly identified
        - **Recall (Non-Diabetic)**: How many actual non-diabetic cases the model correctly identified
        - **F1-Score**: Harmonic mean of precision and recall
        
        In medical diagnostics, high recall for the diabetic class is often prioritized to minimize missed diagnoses.
        """)
        
        # Feature importance (if Random Forest)
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

# Prediction on new input
st.header("Make a New Prediction")
st.markdown("Adjust the sliders below to input values and click 'Predict' to get a diabetes prediction.")

def user_input():
    inputs = {}
    col1, col2 = st.columns(2)
    
    # Divide features between two columns for better layout
    cols = list(X.columns)
    half = len(cols) // 2
    
    with col1:
        for col in cols[:half]:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            inputs[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
    
    with col2:
        for col in cols[half:]:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            inputs[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
            
    return pd.DataFrame([inputs])

# Execute the function to create the input form
new_data = user_input()

# Add prediction button
if st.button("Predict", type="primary"):
    if 'model' not in locals() or not hasattr(model, 'predict'):
        st.error("Please train a model first!")
    else:
        prediction = model.predict(new_data)[0]
        
        # Get prediction probability/confidence if the model supports it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(new_data)[0]
            confidence = proba[int(prediction)]
        else:
            confidence = "Not available for this model"
        
        # Display the prediction result with class label and confidence
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            st.markdown(f"**Class: 1 - Diabetic**")
        else:
            st.markdown(f"**Class: 0 - Non-Diabetic**")
        
        st.markdown(f"**Confidence: {confidence:.2f}**" if isinstance(confidence, float) else f"**Confidence: {confidence}**")
        
        # Add a visual indicator
        if prediction == 1:
            st.error("This individual is predicted to have diabetes.")
        else:
            st.success("This individual is predicted to not have diabetes.")

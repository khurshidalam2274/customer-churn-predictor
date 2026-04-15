import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

sns.set_theme(style='whitegrid')

st.title("Customer Churn Analysis")
st.markdown("Exploratory data analysis and predictive modeling on the Bank Customer Churn dataset.")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
st.header("1. Load Data")

@st.cache_data
def load_data():
    return pd.read_csv("Churn_Modelling.csv")

df = load_data()
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

# ── 2. EDA ────────────────────────────────────────────────────────────────────
st.header("2. Exploratory Data Analysis")

st.subheader("Dataset Info")
st.write(df.describe())

st.subheader("Missing Values")
st.write(df.isnull().sum())

# Churn distribution
st.subheader("Churn Distribution")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
churn_counts = df['Exited'].value_counts()
axes[0].pie(churn_counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Distribution')
sns.countplot(x='Exited', hue='Exited', data=df, palette=['#2ecc71', '#e74c3c'],
              ax=axes[1], legend=False)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Retained', 'Churned'])
axes[1].set_title('Churn Count')
plt.tight_layout()
st.pyplot(fig)

# Churn by Geography and Gender
st.subheader("Churn Rate by Geography and Gender")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
geo_churn = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
geo_churn.plot(kind='bar', ax=axes[0], color='#3498db', edgecolor='black')
axes[0].set_title('Churn Rate by Geography')
axes[0].set_ylabel('Churn Rate')
axes[0].tick_params(axis='x', rotation=0)
gender_churn = df.groupby('Gender')['Exited'].mean()
gender_churn.plot(kind='bar', ax=axes[1], color='#9b59b6', edgecolor='black')
axes[1].set_title('Churn Rate by Gender')
axes[1].set_ylabel('Churn Rate')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
st.pyplot(fig)

# Numerical features distribution
num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
st.subheader("Numerical Features Distribution by Churn")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, feature in enumerate(num_features):
    ax = axes[i // 3][i % 3]
    df[df['Exited'] == 0][feature].hist(ax=ax, alpha=0.5, label='Retained', color='#2ecc71', bins=30)
    df[df['Exited'] == 1][feature].hist(ax=ax, alpha=0.5, label='Churned', color='#e74c3c', bins=30)
    ax.set_title(feature)
    ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[num_features + ['Exited']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
st.pyplot(fig)

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
st.header("3. Preprocessing & Model Training")

@st.cache_data
def preprocess_and_train(df):
    data = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    le = LabelEncoder()
    data['Geography'] = le.fit_transform(data['Geography'])
    data['Gender'] = le.fit_transform(data['Gender'])

    X = data.drop('Exited', axis=1)
    y = data['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

X_train, X_test, y_train, y_test, feature_names = preprocess_and_train(df)

# Model selection
model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])

@st.cache_data
def train_model(model_name, X_train, y_train):
    if model_name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(model_name, X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ── 4. Results ────────────────────────────────────────────────────────────────
st.header("4. Model Results")

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'])
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Confusion Matrix')
plt.tight_layout()
st.pyplot(fig)

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {auc:.3f}')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Feature importance (tree-based models only)
if model_name in ["Random Forest", "Gradient Boosting"]:
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    fi.plot(kind='bar', ax=ax, color='#3498db', edgecolor='black')
    ax.set_title('Feature Importance')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    st.pyplot(fig)

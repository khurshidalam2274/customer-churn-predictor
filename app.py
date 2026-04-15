import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

sns.set_theme(style='whitegrid')

st.title("Customer Churn Predictor")
st.markdown("Exploratory Data Analysis and Churn Prediction using **Artificial Neural Network (ANN)**.")

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

st.subheader("Dataset Statistics")
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
st.header("3. Data Preprocessing")

@st.cache_data
def preprocess(df):
    data = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    le = LabelEncoder()
    data['Geography'] = le.fit_transform(data['Geography'])
    data['Gender']    = le.fit_transform(data['Gender'])
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X.columns.tolist()

X_train, X_test, y_train, y_test, feature_names = preprocess(df)
st.success(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]} | Features: {X_train.shape[1]}")

# ── 4. ANN Configuration ──────────────────────────────────────────────────────
st.header("4. ANN Model Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    hidden_layer_1 = st.slider("Neurons - Layer 1", 32, 256, 64)
with col2:
    hidden_layer_2 = st.slider("Neurons - Layer 2", 16, 128, 32)
with col3:
    max_iter = st.slider("Max Iterations (Epochs)", 50, 500, 200)

activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])

if st.button("Train ANN Model"):
    with st.spinner("Training ANN... please wait"):

        model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_1, hidden_layer_2),
            activation=activation,
            solver='adam',
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            verbose=False
        )
        model.fit(X_train, y_train)

    st.success("Model training complete!")

    # ── 5. Training Loss Curve ────────────────────────────────────────────────
    st.header("5. Training Loss Curve")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(model.loss_curve_, label='Train Loss', color='#e74c3c')
    if model.validation_scores_ is not None:
        ax.plot(model.best_validation_score_ * np.ones(len(model.loss_curve_)),
                linestyle='--', color='#3498db', label=f'Best Val Score: {model.best_validation_score_:.4f}')
    ax.set_title('Loss over Iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # ── 6. Evaluation ─────────────────────────────────────────────────────────
    st.header("6. Model Evaluation")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

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

    # ── 7. Summary ────────────────────────────────────────────────────────────
    st.header("7. Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",    f"{(y_pred == y_test).mean():.2%}")
    col2.metric("AUC Score",   f"{auc:.3f}")
    col3.metric("Iterations",  f"{model.n_iter_}")

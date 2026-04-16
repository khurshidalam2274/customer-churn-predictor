# Customer Churn Predictor

A web app that predicts whether a bank customer will leave (churn) or stay, using an **Artificial Neural Network (ANN)**.

---

## What It Does

Enter a customer's details and the app tells you:
- The **probability** the customer will leave
- A final decision: **WILL CHURN** or **WILL STAY**

---

## Files

| File | Purpose |
|---|---|
| `app.py` | The web app — run this to start |
| `churn_analysis.ipynb` | Data exploration and model analysis |
| `Churn_Modelling.csv` | Dataset with 10,000 bank customers |
| `requirements.txt` | Python packages needed |

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## What is an ANN?

An **Artificial Neural Network (ANN)** is a machine learning model inspired by how the human brain works.

```
Input Layer  →  Hidden Layers  →  Output Layer
(customer       (finds patterns    (churn probability)
 details)        in the data)
```

### How it learns:
1. The ANN takes in customer data (age, balance, credit score, etc.)
2. It passes the data through layers of connected "neurons"
3. Each neuron applies a mathematical function and passes the result forward
4. The final layer outputs a probability between 0 and 1
5. The model learns by comparing its predictions to the real answers and adjusting itself — this is called **backpropagation**

### Why ANN for churn prediction?
- Can detect **complex, non-linear patterns** in customer behavior
- Learns from all 10,000 customer records at once
- Gets better with more data and training

---

## Input Fields

| Field | What to Enter |
|---|---|
| Credit Score | 300 – 850 |
| Geography | France, Germany, or Spain |
| Gender | Male or Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| Number of Products | 1 – 4 |
| Has Credit Card | Yes or No |
| Is Active Member | Yes or No |
| Estimated Salary | Annual salary |

---

## How the App Works

1. Loads 10,000 customer records from the CSV file
2. Preprocesses the data (encodes categories, scales numbers)
3. Trains the ANN model on 80% of the data
4. When you submit a customer's details, the model predicts the churn probability

> The model is trained once and cached, so predictions are fast.

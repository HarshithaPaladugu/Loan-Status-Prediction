# Loan-Status-Prediction
Predicting whether the loan is approved or rejected by considering the necessary parameters of an individual
Thanks for uploading the file `Loan_status.ipynb`. Based on its contents, here is a **README.md** file that summarizes the sequential steps involved in **training a machine learning model and deploying it via a Streamlit application** to predict **loan status**:

---

## 📊 Loan Status Prediction – ML Model & Streamlit App

This project aims to build a **machine learning model** to predict whether a loan should be approved or not, based on applicant and asset details. The model is trained on preprocessed data and deployed using **Streamlit** for interactive user input and prediction.

---

### 📁 Dataset Used

* `loan_approval_dataset_renamed.csv` – contains cleaned and encoded loan application data.
* Target column: `loan_status` (1 = Approved, 0 = Rejected)

---

### 🔁 Project Workflow

#### 1. **Import Libraries**

Essential libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, and `pickle` were imported to perform data manipulation, visualization, model building, and saving.

#### 2. **Load Dataset**

The dataset was loaded using `pandas.read_csv()`, and column names were stripped of whitespace.

```python
df = pd.read_csv('loan_approval_dataset_renamed.csv')
df.columns = df.columns.str.strip()
```

---

#### 3. **Skewness Analysis & Visualization**

Numerical columns were analyzed for skewness, and histograms with KDE plots were generated to understand distribution patterns.

---

#### 4. **Feature & Target Split**

The features (`X`) and target (`y = loan_status`) were separated for modeling.

```python
X = df.drop('loan_status', axis=1)
y = df['loan_status']
```

---

#### 5. **Train-Test Split**

The dataset was split into training and test sets using an 80/20 ratio.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

#### 6. **Model Training – Random Forest**

A `RandomForestClassifier` was trained on the training data.

```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

---

#### 7. **Model Evaluation**

The model was evaluated using metrics such as Accuracy, Precision, Recall, F1-score, ROC-AUC, and a Confusion Matrix.

---

#### 8. **Save the Model**

The trained model was saved using `pickle` for deployment.

```python
with open('loan_status_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
```

---

### 🌐 Streamlit Application

#### 📌 Features:

* Takes 11 user inputs related to the applicant's profile.
* Predicts loan approval status (`Approved` or `Rejected`) using the saved Random Forest model.

#### 🧾 Inputs Taken:

* `no_of_dependents`
* `income_annum`
* `loan_amount`
* `loan_term`
* `cibil_score`
* `residential_assets_value`
* `commercial_assets_value`
* `luxury_assets_value`
* `bank_asset_value`
* `education` (1 = Graduate, 0 = Not Graduate)
* `self_employed` (1 = Yes, 0 = No)

#### ▶️ Run the App:

```bash
streamlit run app.py
```

---


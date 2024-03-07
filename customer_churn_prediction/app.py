from flask import Flask, render_template, request

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from watchdog.events import EVENT_TYPE_OPENED

app = Flask(__name__)

# Load the initial dataset and prepare the models
df = pd.read_excel(r"customer_churn_large_dataset.xlsx")

# # One-hot Encoding
# df = pd.get_dummies(df, columns=['Location'])

# Label encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Remove customer_id, Name column
df2 = df.drop(['CustomerID', 'Name','Location'], axis=1)

# Split dataset into test and train
X = df2.drop('Churn', axis=1)
y = df2['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Standardize features
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Train models
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

clf = LogisticRegression(random_state=1)
clf.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = le.transform([request.form['gender']])[0]
        subscription_length = float(request.form['subscription_length'])
        monthly_bill = float(request.form['monthly_bill'])
        total_usage = float(request.form['total_usage'])
        model_choice = request.form['model_choice']

        # Prepare input for prediction
        input_data = sc_x.transform([[age, gender, subscription_length, monthly_bill, total_usage]])

        # Choose the model
        if model_choice == 'Random Forest':
            model = rfc
        else:
            model = clf

        # Make prediction
        churn_prediction = model.predict(input_data)

        # Generate churn risk scores
        churn_risk_scores = np.round(model.predict_proba(input_data)[:, 1] * 100,2)
        # Churn flag
        if churn_prediction == 1:
            churn_prediction = 'YES'
        else:
            churn_prediction = 'NO'

        return render_template('result.html', churn_prediction=churn_prediction,churn_risk_scores=churn_risk_scores, inputs=request.form)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

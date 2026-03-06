import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("employee_data.csv")

# Encode text data
encoder = LabelEncoder()
data['skills'] = encoder.fit_transform(data['skills'])
data['education'] = encoder.fit_transform(data['education'])

# Drop unnecessary columns
data = data.drop(['first_name','last_name','emp_id'], axis=1)

# Features and target
X = data.drop('salary', axis=1)
y = data['salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train,y_train)

# Save model
pickle.dump(model, open("salary_model.pkl","wb"))

print("Model trained and saved!")
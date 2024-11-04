# Import Libraries 
import pandas as pd ;
import numpy as np ; 
import matplotlib.pyplot as plt ; 
import seaborn as sns  ; 
# Load Data
df = pd.read_csv('Advertising.csv') 
# view the first few rows
print(df.head()) 
# check for data types and null values
print(df.info())  
# get summary statistics
print(df.describe())  
# Check for Missing Values
print(df.isnull().sum()) 
# Visualize the Data Distribution 
df.hist(bins=30, figsize=(10, 8))
plt.show()
# Visualize Relationships 
sns.pairplot(df)
plt.show()
# Correlation Matrix
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()
'''Data Preprocessing :  preprocess it for modeling. This step ensures that the data is in the best form for the machine learning model
-- the most correclated is TV with 0.78'''
# Splitting the Data 
from sklearn.model_selection import train_test_split
 # Input features
X = df[['TV', 'Radio', 'Newspaper']] 
# Target variable
y = df['Sales']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature Scaling : step standardizes the data to have a mean of 0 and a standard deviation of 1. 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Linear Regression Model 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# Model Evaluation 
y_pred = model.predict(X_test)
'''MAE : measures the average magnitude of errors in predictions.
MSE : gives more weight to larger errors, penalizing significant deviations.
R-squared : indicates how much of the variance in the target variable is explained by the model. Closer to 1 is better.'''
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MAE:", mean_absolute_error(y_test, y_pred)) # 1.46 
print("MSE:", mean_squared_error(y_test, y_pred)) # 3.1741 
print("R-squared:", r2_score(y_test, y_pred)) # 89.94% 
# model seems to perform quite well ! 
# Plot Predicted Sales
plt.plot(y_pred, label="Predicted Sales", color="blue")
plt.xlabel("Data Point")
plt.ylabel("Sales Prediction")
plt.title("Predicted Sales")
plt.legend()
plt.show()
# Visualize Predictions vs. Actual 
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

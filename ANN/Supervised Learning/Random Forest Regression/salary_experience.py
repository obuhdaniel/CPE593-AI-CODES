import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

'''
Generate Synthetic Data set
'''

np.random.seed(42)
experience = np.random.uniform(1,10,50)
salary = 2000 * experience + np.random.normal(0, 5000, 50) 

data = pd.DataFrame({
    'Years_of_experience': experience,
    'Salary': salary
})

'''
Step 2: EDA
'''

print("First five rows of the dataset:\n", data.head()) 

plt.figure(figsize= (8,5))
sns.scatterplot(x=data['Years_of_experience'], y=data['Salary']) 
plt.xlabel("Years of Experience") 
plt.ylabel("Salary ($)") 
plt.title("Scatter Plot: Years of Experience vs. Salary") 
plt.show() 

'''
Step 3: Split Dataset
'''

X= data[['Years_of_experience']]
y= data[['Salary']]

#Divide the testing Datasets into Training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Step 4: Train the Linear Regression Model
'''
model = LinearRegression()
model.fit(X_train, y_train)

'''
Step 5: Model Evaluation
'''
y_pred = model.predict(X_test)

#model eval
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error 
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error 
rmse = np.sqrt(mse)  # Root Mean Squared Error 
r2 = r2_score(y_test, y_pred)  # R-squared score 

print(f"Mean Absolute Error (MAE): {mae:.2f}") 
print(f"Mean Squared Error (MSE): {mse:.2f}") 
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}") 
print(f"R-squared Score (R²): {r2:.2f}") 

'''
Step 6: Visualization Of Regression Line
'''
plt.figure(figsize=(8,5))
sns.scatterplot(x=X_test['Years_of_experience'], y=y_test, label="Actual") 
sns.lineplot(x=X_test['Years_of_experience'], y=y_pred, color='red', label="Predicted") 
plt.xlabel("Years of Experience") 
plt.ylabel("Salary ($)") 
plt.title("Linear Regression: Salary Prediction") 
plt.legend() 
plt.show()
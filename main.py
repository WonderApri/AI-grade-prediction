import numpy as np #np is the short form for numpy 
import pandas as pd 
from sklearn.model_selection import train_test_split #one of Pythons main ML libraries to do linear regression, decision trees etc #train_test_split is used to split data (because then MLM would just memorise all your data) into training and testing sets 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt 

#Now we prepare the Data 
data = { 
    'Hour_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
    'score': [11, 25, 32, 40, 59, 62, 69, 78, 82, 89] 
    } 

df = pd.DataFrame(data) #Pandas df makes a table, DataFrame is the table like structure similar to excel sheet, data is the raw data given 
x=df[['Hour_studied']] #independent variable (input) 
y=df['Score'] #dependent variable (output) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 
#This is training the Model, splitting it into training and testing, randoming picking some to learn patterns from and others to check how well it learned 
#test_size=0.2 means 20% of data is for testing, 80% for training 
#random_state=42 is just a seed for random number generator to ensure reproducibility 
#reproducibility means every time you run the code, you get the same results

model = LinearRegression() #Creating the model using Linear Regression 


model.fit(x_train, y_train) #Fitting the model with training data 
y_pred = model.predict(x_test) #Predicting the scores for test data 
print("Predicted scores:", y_pred) #Evaluating the Model 
mse = mean_squared_error(y_test, y_pred) #Calculating Mean Squared Error to see how well the model performed 
print("Mean Squared Error:", mse) #y_test are the actual scores, y_pred are the predicted scores by the model 

#Visualizing the results 
x_1d = x['Hours_studied'].values 
y_pred_full = model.predict(x).flatten() 
#line of best fit through data to regress it 


plt.scatter(x_1d, y, color='blue', label='Actual Scores') #Scatter plot of actual scores 
plt.plot(x_1d, y_pred_full, color='red', label='Predicted Regression Line') #Regression line 
#scatter is for points, plot is for lines 


plt.xlabel('Hours Studied') 
plt.ylabel('Scores') 
plt.title('Hours Studied vs Scores Prediction') 
plt.legend() 
plt.show()
import pandas as pd
import math 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

df= pd.read_csv('input.csv')
print(df.head())

X = df['total study hrs'].tolist()
y = df['total marks obtained'].tolist()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# print("X_train:",X_train)
# print("X_test:",X_test)
# print("y_train:",y_train)
# print("y_test:",y_test)

# Mean Function
def mean(values):
    return sum(values) / len(values)

# Numerator 
def covariance(x, x_mean, y, y_mean):
    total = 0                      
    # Loop har index ke liye
    for i in range(len(x)):
        diff_x = x[i] - x_mean     
        diff_y = y[i] - y_mean     

        product = diff_x * diff_y  
        total += product        

    return total
      
# Denominator
def variance(values, mean_val):
    var=[]
    for x in values:
        diff = x - mean_val
        sqr = diff**2
        var.append(sqr)

    return sum(var)           

def coefficients(X, y):
    x_mean = mean(X) #function call of mean
    y_mean = mean(y)
    m = covariance(X, x_mean, y, y_mean) / variance(X_train, x_mean) #slope
    c =  round(y_mean - m * x_mean,2) #intercept
    return c, m  


def predict(X, c, m):
    predictions = []
    for x in X:
        y_pred = c + m * x
        predi =round(y_pred,2)
        predictions.append(predi)
    return predictions


def mse_metric(actual, predicted):
    error = 0.0
    for i in range(len(actual)):
        error += (predicted[i] - actual[i]) ** 2
    return round((error / len(actual)),2)

def rmse_metric(actual, predicted):
    error = 0.0
    for i in range(len(actual)):
        error += (predicted[i] - actual[i]) ** 2
        rmse=math.sqrt(error / len(actual))
        
    return round(rmse,2)

# Coefficients
c, m = coefficients(X_train, y_train)
print("Slope (m):", m)
print("Intercept (c):", c)

# Predictions
predicted = predict(X_test, c, m)
print("Predicted values:", predicted)

result_df = pd.DataFrame({
    'Sleep Hours ': X_test,
    'Actual Marks ': y_test,
    'Predicted Marks': predicted
})

result_df.to_csv('predicted_output.csv', index=False)

print(result_df.head())


# MSE
mse = mse_metric(y_test, predicted)
print("MSE:", mse)

# RMSE
rmse = rmse_metric(y_test, predicted)
print("RMSE:", rmse)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Training Data')
plt.plot(X_test, predicted, color='red', linewidth=2, label='Regression Line')  
plt.xlabel("X (Input)")
plt.ylabel("y (Output)")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
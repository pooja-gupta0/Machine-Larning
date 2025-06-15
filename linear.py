# Sample full data   
X = [15, 20, 35, 45, 50, 65, 70, 85, 90]
y = [5,13,11,27,32,39,43,51,54]

# Split into train and test
X_train = [10, 25, 30, 45]
y_train = [5,13,11,27]

X_test = [50,65,70,85,90]
y_test = [32,39,43,51,54]

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
    x_mean = mean(X) #function call of mean for x
    y_mean = mean(y) #function call of mean for y
    m = covariance(X, x_mean, y, y_mean) / variance(X, x_mean) #slope  function call
    c =  round(y_mean - m * x_mean,2) #intercept
    return c, m  


def predict(X, c, m):
    predictions = []
    for x in X:
        y_pred =  m * x + c
        predi =round(y_pred,2)
        predictions.append(predi)
    return predictions

import math

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
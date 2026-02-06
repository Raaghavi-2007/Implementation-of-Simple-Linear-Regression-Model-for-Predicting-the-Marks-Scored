# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries. 
2. Store the independent variable (hours studied) and the dependent variable (marks obtained) in arrays and reshape the input data for model training.
3. Create and train the Linear Regression model using least squares.
4. Predict the marks using the input taken from the user.
5. Finally, display the predicted marks and  a graph showing the data points and the regression line for visualization.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Raaghavi S
RegisterNumber: 25012715
*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

Y = np.array([38, 42, 55, 60, 70, 79])

model = LinearRegression()

model.fit(X, Y)

hours = float(input("Enter number of hours studied: "))

predicted_marks = model.predict([[hours]])

print("Predicted Marks Scored:", round(predicted_marks[0], 2))

plt.scatter(X, Y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression – Marks Prediction")
plt.show()

```

## Output:
<img width="877" height="642" alt="image" src="https://github.com/user-attachments/assets/1df28426-2745-4390-bdc4-30ed5e94030c" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

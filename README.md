# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Imports the required libraries. 
2. Store the independent variable (hours studied) and the dependent variable (marks obtained) in arrays and reshape the input data for model training.
3. Create and train the Linear Regression model using least squares.
4. Predict the marks using the input taken from the user.
5. Finally, the predicted marks are displayed and a graph is plotted showing the data points and the regression line for visualization.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

Y = np.array([78, 56, 63, 89, 91, 86])

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
<img width="936" height="648" alt="Screenshot 2026-01-30 113108" src="https://github.com/user-attachments/assets/234d8422-cdc2-4bfe-aeae-bbdc08f7707c" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

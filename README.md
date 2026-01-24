# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
Enter number of hours studied: 6
Predicted Marks Scored: 89.38

<img width="563" height="453" alt="ml-2" src="https://github.com/user-attachments/assets/23c6ac56-4a1c-454e-9a6d-2938aa11ecb5" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

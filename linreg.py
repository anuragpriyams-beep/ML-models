import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
#I will apply methods of linear regression like formula, gd, sgd

#dataset creation

x = np.random.rand(1000,1)
y = 4 + 3*x + np.random.randn(1000,1)*2

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#stochastic grad descent 

epochs = 10000
t1 = 0
t2 = 0
eta = 0.015
err = np.zeros(epochs)
for j in range(epochs):
    i = np.random.choice(np.arange(799))
    pred = t1 + t2*X_train[i]
    err[j] = np.mean(0.5*(pred - y_train[i])**2)
    t1 = t1 - eta * (pred - y_train[i])
    t2 = t2 - eta * ((pred - y_train[i])*X_train[i])

y_pred = t1 + t2*X_test

# Create subplots: 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Predictions vs Actual
ax1.scatter(X_test, y_test, alpha=0.5, label='actual')
ax1.plot(X_test, y_pred, color='red', linewidth=2, label='predicted')
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Predictions vs Actual')

# Plot 2: Error over Epochs
ax2.plot(err, linewidth=2, color='blue')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Error (MSE)')
ax2.set_title('Training Error over Epochs')
ax2.grid(True)

plt.tight_layout()
plt.show()
print(t1, ' ,' , t2)

#Why np.mean is used
#
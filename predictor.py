import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.array([[10, 4], [30, 6], [80, 8], [15,8], [100, 5]])
y_train = np.array([5, 35, 120, 40,90])

model = LinearRegression()
model.fit(X_train, y_train)

new_movie = np.array([[50, 9]])
predicted_earnings = model.predict(new_movie)

print(f"Predicted Box Office Earnings: {predicted_earnings[0]:.2f} Crores")

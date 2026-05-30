import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([2,3,5,6,8]).reshape(-1,1)
y-np.array([50,60,65,70,80])

model=LinearRegression()
model.fit(x,y)
print("Regression Equation:y={model.coef_[0]:.2f}X+{model.intercept:.2f}")

try:
    hours=float(input("Enter hours:"))
    predicted_value=model.predict(np.array([[hours]]))
    print("Predicted value:{predicted_value[0]:.2f}")
except ValueError:
    print("Invalid Input!\n Please Enter a numeric value")
    exit()
plt.scatter(x,y,color="blue",lable="Original data")
plt.plot(x,model.predict(x),color="red",label="Regression")
plt.scatter([hours],predicted_value,color="voilet",label="Predicted point",s=100)
plt.text(hours,predicted_value,"({predicted_value[0]:.2f)"}))

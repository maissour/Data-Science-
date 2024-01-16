import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('C:/Users/Ahcene/Desktop/Project/Data/Battery_RUL_cleaned.csv')

# Extract features and RUL values
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

# Normalize the data 
mms_x = MinMaxScaler()
x_transfromed=mms_x.fit_transform(x)

# Split the data 
xtrain,xtest,ytrain,ytest=train_test_split(x_transfromed,y,test_size=0.2,random_state=0)

# Model training
model = RandomForestRegressor(n_estimators =200, random_state = 42)

model.fit(xtrain, ytrain)

# Predict using the machine learning model
predicted_rul_ml = model.predict(xtest)

print("Mean squared error: %.2f" % mean_squared_error(ytest, predicted_rul_ml))
print("R2 score: %.2f" % r2_score(ytest, predicted_rul_ml))



m,c=np.polyfit(ytest, predicted_rul_ml,1)
abline=[m*i+c for i in ytest]

# Plot the results
plt.scatter(ytest, predicted_rul_ml, label='RF Predictions')

plt.plot(ytest, abline,'red',label='Fuzzy Adjusted Predictions')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.legend()
plt.show()

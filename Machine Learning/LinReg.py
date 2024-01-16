import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression

#import data
df = pd.read_csv('path/to/Battery_RUL_cleaned.csv')


x=df.iloc[:,:-1]
y=df.iloc[:,-1]

# Normalize the data 
mms_x = MinMaxScaler()
x_transfromed=mms_x.fit_transform(x)

# Split the data 
xtrain,xtest,ytrain,ytest=train_test_split(x_transfromed,y,test_size=0.2,random_state=42)

# Model training
clf = LinearRegression()
clf.fit(xtrain, ytrain)


# Make prediction
ypred=clf.predict(xtest)

import numpy as np

m,c=np.polyfit(ytest, ypred,1)

abline=[m*i+c for i in ytest]
# Calculate the MSE
print("Mean squared error: %.2f" % mean_squared_error(ytest, ypred))
print("mean absolute error: %.2f" % mean_absolute_error(ytest, ypred))
print("R2 score: %.2f" % r2_score(ytest, ypred))

# Visualisation des résultats de la prédiction

plt.figure()
plt.scatter(ytest,ypred)
plt.plot(ytest, abline,'black')
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.title('RUL')
plt.show()




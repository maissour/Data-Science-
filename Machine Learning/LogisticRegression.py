import pandas as pd
from sklearn.metrics import confusion_matrix ,accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#import data
df = pd.read_csv('path/to/heart.csv',delimiter=',')
df.head()

x = df.iloc[:,0:13]
y = df.iloc[:,-1]

#splite data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

#Model
logistic_regression = LogisticRegression(solver='liblinear', max_iter=100)

# train the model 
logistic_regression.fit(x_train,y_train)

#predict 
y_pred_logistic = logistic_regression.predict(x_test)

#accuracy
accuracy = accuracy_score(y_test,y_pred_logistic)
accuracy_percentage = 100 * accuracy
print("accuracy ...")
print(accuracy_percentage)
print("Matrice de confusion ...")  
print(confusion_matrix(y_test,y_pred_logistic)) 

#Plot Model predict
plt.scatter(x_train.iloc[:,0][y==0],x_train.iloc[:,3][y==0],color='b',label='c1')
plt.scatter(x_train.iloc[:,0][y==1],x_train.iloc[:,3][y==1],color='r',label='c2')
plt.scatter(x_test.iloc[:,0],x_test.iloc[:,4],c=y_pred_logistic,s=150,edgecolors='g',marker='X')
plt.xlabel('Age')
plt.ylabel('trestbps')
plt.title('logistic regression',fontweight='bold')
plt.legend()

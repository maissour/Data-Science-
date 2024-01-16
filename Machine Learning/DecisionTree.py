import pandas as pd
from sklearn.metrics import confusion_matrix ,accuracy_score 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree  
import pydotplus
from io import StringIO

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
#import data
df = pd.read_csv('D:/Project/heart.csv',delimiter=',')
df.head()

x = df.iloc[:,0:13]
y = df.iloc[:,-1]

#splite data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.30)

#Model
decision_Tree=DecisionTreeClassifier()


# train the model 
decision_Tree.fit(x_train,y_train)


#predict 
y_pred_decision_tree=decision_Tree.predict(x_test)

#


#accuracy
accuracy = accuracy_score(y_test,y_pred_decision_tree)
accuracy_percentage = 100 * accuracy
print("accuracy ...")
print(accuracy_percentage)
print("Matrice de confusion ...")  
print(confusion_matrix(y_test,y_pred_decision_tree)) 

plt.scatter(x_train.iloc[:,0][y==0],x_train.iloc[:,3][y==0],color='b',label='c1')
plt.scatter(x_train.iloc[:,0][y==1],x_train.iloc[:,3][y==1],color='r',label='c2')
plt.scatter(x_test.iloc[:,0],x_test.iloc[:,4],c=y_pred_decision_tree,s=150,edgecolors='g',marker='X')
plt.xlabel('Age')
plt.ylabel('trestbps')
plt.title('Decision Tree',fontweight='bold')
plt.legend()

#draw the tree model
dotfile = StringIO()
tree.export_graphviz(decision_Tree, out_file=dotfile)
graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png("dtree.png")
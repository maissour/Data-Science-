from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import os
import cv2
from tqdm import tqdm


features = []

for i in range(1,41):
    loc0='path/to/faces-test/s'+str(i)
    for img in tqdm(os.listdir(loc0)):
       f = cv2.imread(os.path.join(loc0,img))
       fr = cv2.resize(f,(64,64))
       features.append(fr)

labels = []
for i in range(1,41):
    for img in tqdm(os.listdir(loc0)):
        labels.append(i-1)
        
x = np.array(features)
y=np.array(labels)
Y=y.reshape(-1,1)

print(x.shape)
print(Y.shape)
    
target_names = []
for i in range(1,41):
	name=''
	name+='s'+str(i)
	target_names.append(name)

X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.10)

#

ytrain_h = to_categorical(y_train,num_classes =40 ) # 40 distinct people
ytest_h = to_categorical(y_test,num_classes = 40)
#Define a CNN model to make predictions


model = Sequential()

model.add(Conv2D(filters = 20, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape=x.shape[1:]))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 50, kernel_size = (6,6),padding = 'Same', 
                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 150, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,3)))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(40, activation = "softmax"))
model.summary()

#Compile the model

model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy']) 

epoch=40
history=model.fit(X_train,ytrain_h,epochs=epoch,validation_data=(X_test,ytest_h))

print(model.evaluate(X_train,ytrain_h))
print(model.evaluate(X_test,ytest_h))


preds = model.predict(X_test)
y_pred = model.predict_classes(X_test)


y_test = y_test.reshape(-1,)

diff = y_test - y_pred
diff = diff.reshape(-1,1)

true = 0
for i in range(0,len(diff)):
    if diff[i] == 0:
         true = true + 1

Cnn_accuracy = round(100*true/len(diff),2)
print("Cnn_accuracy is %", Cnn_accuracy)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_pred, labels=range(40)),annot = True, cmap = "rainbow")
plt.title("Confusion Matrix")
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
print("Classification report:")
print(classification_report(y_test, y_pred,labels=range(40)))

# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


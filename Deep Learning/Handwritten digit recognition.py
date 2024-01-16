# Import the required library
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,Sequential
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST data
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()

# Normalize the X data 
xtrain=xtrain/255
xtest=xtest/255

# CNN Model
model=Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),
                        activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(xtrain,ytrain,validation_data=(xtest,ytest),
                  epochs=10,batch_size=32)

# Evaluation
l,s=model.evaluate(xtest,ytest)

# Prediction
y_pre=model.predict(xtest)
y_class=[np.argmax(i) for i in y_pre]
print(y_class[:5])
print(ytest[:5])

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
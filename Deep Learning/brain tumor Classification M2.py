import tensorflow as tf
import numpy as np
import splitfolders
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.models import Sequential


# train, test split
splitfolders.ratio('D:/Project/Artificial Intelligence/cnn code/brain_tumor_dataset/'
                   , output="D:/Project/Artificial Intelligence/cnn code/brain_tumor_dataset_split"
                   , ratio=(0.8, 0.2))

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

path='D:/Project/Artificial Intelligence/cnn code/brain_tumor_dataset_split/train/'
training_set = train_datagen.flow_from_directory(path,
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('D:/Project/Artificial Intelligence/cnn code/brain_tumor_dataset_split/val/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#. 
cnn =Sequential()
cnn.add(Conv2D(filters=32,kernel_size=5, activation='relu', input_shape=[64, 64, 3]))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(Dropout(0.1))

cnn.add(Conv2D(filters=64,kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(Dropout(0.1))

cnn.add(Conv2D(filters=128,kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2))
cnn.add(Dropout(0.1))

cnn.add(Flatten())
cnn.add(Dense(units=256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=1, activation='sigmoid'))
cnn.summary()

#

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 1, 
                                        restore_best_weights = True)
epochs=10

history=cnn.fit(x = training_set, validation_data = test_set, epochs =epochs)

print('accuracy training:',cnn.evaluate(training_set))
print('accuracy test:',cnn.evaluate(test_set))

#

X_test, Y_test= next(test_set)
print('ytest shape is:',Y_test.shape)

pred=cnn.predict_classes(X_test)
#
pred=tf.reshape(pred, (32,))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
sns.heatmap(confusion_matrix(Y_test,pred), annot = True, cmap = "rainbow")
plt.title("Confusion Matrix")
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
print("Test Accuracy:",accuracy_score(Y_test,pred))
print("Classification report:")
print(classification_report(Y_test,pred))

#
test_image = image.load_img('D:/Project/Artificial Intelligence/BTY2.jpg', target_size = (64, 64))
plt.imshow(test_image)
plt.title('Test Brain Image'), plt.xticks([]), plt.yticks([])
plt.show()

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'brain tumor present'
else:
    prediction = 'no brain tumor'
print(prediction)

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


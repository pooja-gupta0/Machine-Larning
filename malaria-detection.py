!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d nipunarora8/malaria-detection-dataset

import zipfile
zip_ref = zipfile.ZipFile('/content/malaria-detection-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/Dataset/Train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = keras.utils.image_dataset_from_directory(
    directory='/content/Dataset/Test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="validation",
    seed=123
)


#normalize
def process(image,label):
  image = tf.cast(image/255. ,tf.float32)
  return image,label

train_ds = train_ds.map(process)
val_ds = val_ds.map(process)

# Create CNN model

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))


model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=10,validation_data=val_ds)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

import cv2

test_img = cv2.imread('/content/yes.jpg')

plt.imshow(test_img)

test_img.shape

test_img  = cv2.resize(test_img,(256,256))

test_input = test_img.reshape((1,256,256,3))

predict = model.predict(test_input)

predict

import numpy as np

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds = np.round(preds).astype(int).flatten()  # Binary classification (0 or 1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())


from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Print the classification report
print("\nClassification Report:\n", classification_report(y_true, y_pred))

#print accuary score
print("\nAccuracy Score:",round(accuracy_score(y_true, y_pred),2))

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()

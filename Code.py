from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow import keras
import cv2
import numpy as np

model = Sequential()
model.add(Conv2D(64, (3, 3), activation = 'relu', input_dim = (150, 150, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(30, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(6450, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(4, activation = 'linear'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mse', metrics = ['mae'])

model.summary()

train_labels = []
train_images = []
for i in range(1, 16):
  img = cv2.imread(images_path + f'/{i}.jpeg')
  img = cv2.resize(img, (150, 150))
  train_images.append(img)
  with open(labels_path + f'/{i}.txt', 'r') as file:
    content = (file.read()).split()[1:]
  content = [float(j) for j in content]
  train_labels.append(content)
val_labels = []
val_images = []
for i in range(16, 25):
  img = cv2.imread(images_path + f'/{i}.jpeg')
  img = cv2.resize(img, (150, 150))
  val_images.append(img)
  with open(labels_path + f'/{i}.txt', 'r') as file:
    content = (file.read()).split()[1:]
  content = [float(j) for j in content]
  val_labels.append(content)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

train_images = np.array(train_images) / 255
val_images = np.array(val_images) / 255

history = model.fit(train_images, train_labels, epochs = 50, validation_data = (val_images, val_labels))

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']
train_loss = [1 if i >= 1 else i for i in train_loss]
val_loss = [1 if i >= 1 else i for i in val_loss]

import matplotlib.pyplot as plt
epochs = range(1, len(train_loss) + 1)

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


predict_img = cv2.imread(images_path + '/images.jpeg')
predict_img = cv2.resize(predict_img, (150, 150))
print(predict_img.shape)

predict_img = np.array(predict_img) / 255
predict_img = np.expand_dims(predict_img, axis = 0)
predict_img.shape

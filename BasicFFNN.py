import numpy as np
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

### Data Pre-processing ###

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

### Model Architecture ###

model = models.Sequential()
model.add(layers.Input(shape= (28*28, )))
model.add(layers.Dense(512, activation= 'relu'))
model.add(layers.Dense(256, activation= 'relu'))
model.add(layers.Dense(10, activation= 'softmax'))

### Optimizer and Loss ###

model.compile(optimizer= 'rmsprop', loss= 'categorical_crossentropy', metrics= ['accuracy'])

### Training and Evaluation ### 


model.fit(train_images, train_labels, epochs= 5, batch_size= 128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test loss = {test_loss}, Test accuracy = {test_acc}")

model.summary()

from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras import layers

### Data Pre-processing ###

max_features = 10000
maxlen = 500


(seq_train, y_train), (seq_test, y_test) = imdb.load_data(num_words= max_features)

seq_train = pad_sequences(seq_train, maxlen= maxlen)
seq_test = pad_sequences(seq_test, maxlen= maxlen)

### Model Architecture ###

model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.LSTM(32, return_sequences= True))
model.add(layers.LSTM(32, return_sequences= True))
model.add(layers.LSTM(32, return_sequences= False))
model.add(layers.Dense(1, activation= 'sigmoid'))

### Optimizer and Loss ###

model.compile(optimizer= 'rmsprop', loss= 'binary_crossentropy', metrics= ['acc'])

### Training and Evaluation ### 

model.fit(seq_train, y_train, epochs= 10, batch_size= 128)
test_loss, test_acc = model.evaluate(seq_test, y_test)
print(f"Test loss = {test_loss}, Test accuracy = {test_acc}")

model.summary()
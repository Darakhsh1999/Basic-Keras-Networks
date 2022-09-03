from keras import Model
from keras import layers
from keras import Input

### Define layer flow architecture ###
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation= 'relu')(input_tensor)
x = layers.Dense(32, activation= 'relu')(x)
output_tensor = layers.Dense(10, activation= 'softmax')(x)

### Create the model ### 
model = Model(input_tensor, output_tensor)

model.summary()
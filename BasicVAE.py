import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from KerasDataSet import VAE_data
import numpy as np

# Parameters
img_shape = (28,28,1)
batch_size = 16
laten_dim = 2

def Sampling(args):

    ''' Samples the laten variable from the encoder network '''

    z_mean, z_log_var = args
    epsilon = K.random_normal(shape= (K.shape(z_mean)[0], laten_dim), mean= 0.0, stddev= 1.0)

    return z_mean + K.exp(z_log_var) * epsilon
    
class CustomVAE(keras.layers.Layer):
    
    def VAE_loss(self, x, z_decoded):

        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        KL_loss = -5e-4 * K.mean(1+ z_log_var - K.square(z_mean) - K.exp(z_log_var), axis= 1)

        return K.mean(xent_loss + KL_loss)

    def __call__(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.VAE_loss(x, z_decoded)
        self.add_loss(loss, inputs= inputs)
        return x


# Encoder
input_image = layers.Input(shape= img_shape)
x = layers.Conv2D(32, 3, padding= 'same', activation= 'relu')(input_image)
x = layers.Conv2D(64, 3, padding= 'same', activation= 'relu', strides= (2,2))(x)
x = layers.Conv2D(64, 3, padding= 'same', activation= 'relu')(x)
x = layers.Conv2D(64, 3, padding= 'same', activation= 'relu')(x)
conv_shape = K.int_shape(x) # Shape of data after convolutional part
x = layers.Flatten()(x)
x = layers.Dense(32, activation= 'relu')(x)

# Output variables
z_mean = layers.Dense(laten_dim)(x)
z_log_var = layers.Dense(laten_dim)(x)

z = layers.Lambda(Sampling)([z_mean, z_log_var]) # laten variable


# Decoder
decoder_input = layers.Input(K.int_shape(z)[1:])
x = layers.Dense(np.prod(conv_shape[1:]), activation= 'relu')(decoder_input)
x = layers.Reshape(conv_shape[1:])(x)
x = layers.Conv2DTranspose(32, 3, padding= 'same', activation= 'relu', strides= (2,2))(x)
x = layers.Conv2D(1, 3, padding= 'same', activation= 'sigmoid')(x)
decoder = Model(decoder_input, x)

z_decoded = decoder(z) # latent variable -> constructed image
custom_vae = CustomVAE() 
y = custom_vae([input_image, z_decoded]) # Custom layer for custom loss



### Data Pre-processing ###
x_train, x_test = VAE_data()

### Model Architecture ###
vae = Model(input_image, y)

### Optimizer and Loss ###
vae.compile(optimizer= 'rmsprop')

### Training ### 

vae.fit(x= x_train, y= None, 
        shuffle= True, 
        epochs= 10,
        batch_size= batch_size,
        validation_data= (x_test, None))


vae.summary()


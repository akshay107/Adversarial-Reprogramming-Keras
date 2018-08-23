import numpy as np
from keras.applications import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.datasets import mnist
import keras.backend as K
import cv2
import keras
from keras.layers import ZeroPadding2D, Input
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers,regularizers
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import math

K.clear_session()
K.set_learning_phase(0)

print("Loading Inception model")
inception_model = InceptionV3()
#W_before = inception_model.get_weights()
## Preparing MNIST dataset
# input image dimensions
print("Preparing MNIST dataset")
batch_size = 32
epochs = 20
img_rows, img_cols = 28, 28
num_classes = 1000

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 3)


X_train = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2],3)).astype('uint8')
X_test = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2],3)).astype('uint8')

for i in range(x_train.shape[0]):
	X_train[i] = cv2.cvtColor(x_train[i,:,:,0],cv2.COLOR_GRAY2RGB)

for i in range(x_test.shape[0]):
	X_test[i] = cv2.cvtColor(x_test[i,:,:,0],cv2.COLOR_GRAY2RGB)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
## MNIST dataset prepared

# Masking matrix
print("Preparing Masking Matrix")
M = np.ones((299,299,3)).astype('float32')
M[135:163,135:163,:] = 0

# Adverserial Reprogramming layer
class MyLayer(Layer):
    def __init__(self, W_regularizer=0.05, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.l2(W_regularizer)
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert len(input_shape) == 4
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(299,299,3),
                                      initializer=self.init, regularizer=self.W_regularizer,
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        prog = K.tanh(self.W*M)
        out = x + prog
        return out
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.05
	decay = 0.96
	epochs_drop = 2.0
	lrate = initial_lrate * (1./(1. + decay*math.floor(epoch/epochs_drop)))
        print("Epoch = %d learning rate = %f"%(epoch,lrate))
	return lrate

x = Input(shape=input_shape)
x_aug = ZeroPadding2D(padding=((135,136),(135,136)))(x)
out = MyLayer()(x_aug)
probs = inception_model(out)

model = Model(inputs=x,outputs=probs)

# Freezing InceptionV3 model
model.layers[-1].trainable = False

print(model.summary())

adam = Adam(lr=0.05,decay=0.48)
model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

# learning schedule callback
#lrate = LearningRateScheduler(step_decay)
#callbacks_list = [lrate]

#model.fit(X_train, y_train, batch_size=batch_size, callbacks=callbacks_list, epochs=epochs, verbose=1)
#model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(X_train[:100],y_train[:100]))

#model.load_weights('trial.h5')
#score = model.evaluate(X_train[:100], y_train[:100], verbose=0)
#print('Train loss:', score[0])
#print('Train accuracy:', score[1])
#model.save_weights('trial.h5')
model.save_weights('adversarial.h5')


#model.load_weights('adversarial.h5')


score = model.evaluate(X_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

#W_after = inception_model.get_weights()


#for i in range(len(W_after)):
#	if not np.array_equal(W_after[i],W_before[i]):
#		print("Fuck")
#		print(i)
#model.load_weights('adversarial.h5')

#score = model.evaluate(X_train, y_train, verbose=0)
#print('Train loss:', score[0])
#print('Train accuracy:', score[1])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#model.save_weights('adversarial.h5')



'''
data = np.concatenate((X_train,X_test),axis=0)
label = np.concatenate((y_train,y_test),axis=0)

pred = model.predict(data)

top_probs = np.zeros((10))

top_probs_idx = np.zeros((10))

for i in range(len(data)):
     if np.argmax(label[i])==np.argmax(pred[i]) and pred[i][np.argmax(pred[i])]>=top_probs[np.argmax(pred[i])]:
             top_probs[np.argmax(pred[i])] = pred[i][np.argmax(pred[i])]
             top_probs_idx[np.argmax(pred[i])] = i

interim_model = Model(inputs=model.input,outputs=model.layers[-2].output)
#interim_model.summary()


from PIL import Image

imgs = np.load("imgs.npy")

for i in range(10):
	fig = np.around((imgs[i] + 1.0) / 2.0 * 255)
	fig = fig.astype(np.uint8).squeeze()
	pic = Image.fromarray(fig)
	pic.save("%d_new.png"%i)


org_img = load_img("6_new.png")
np_img = img_to_array(org_img)
np_img = np.expand_dims(np_img, axis=0)
final_img = preprocess_input(np_img)
pred = inception_model.predict(final_img)


for i in range(10):
	org_img = load_img("%d_new.png"%i)
	np_img = img_to_array(org_img)
	np_img = np.expand_dims(np_img, axis=0)
	final_img = preprocess_input(np_img)
	pred = inception_model.predict(final_img)
	label = decode_predictions(pred)
	print(np.argmax(pred),np.max(pred), label[0][0][1])

(0, 0.6044866, u'tench')
(1, 0.87796897, u'goldfish')
(2, 0.6075824, u'great_white_shark')
(3, 0.6018741, u'tiger_shark')
(4, 0.9041798, u'hammerhead')
(5, 0.91017926, u'electric_ray')
(6, 0.95157474, u'stingray')
(7, 0.8722351, u'cock')
(8, 0.8244661, u'hen')
(9, 0.67793137, u'ostrich')


imgs = np.load("imgs.npy")
pred = inception_model.predict(imgs)

for i in range(10):
     print(np.argmax(pred[i]),np.max(pred[i]))
 
(0, 0.9732293)
(1, 0.9910329)
(2, 0.95925266)
(3, 0.9316272)
(4, 0.9861369)
(5, 0.9819935)
(6, 0.9761731)
(7, 0.9852461)
(8, 0.96967906)
(9, 0.9955556)
'''

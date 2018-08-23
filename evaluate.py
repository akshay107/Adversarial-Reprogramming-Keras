import numpy as np
from keras.applications import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

inception_model = InceptionV3()


for i in range(10):
	org_img = load_img("%d_new.png"%i)
	np_img = img_to_array(org_img)
	np_img = np.expand_dims(np_img, axis=0)
	final_img = preprocess_input(np_img)
	pred = inception_model.predict(final_img)
	label = decode_predictions(pred)
	print(np.argmax(pred),np.max(pred), label[0][0][1])

## Gives the following output using trained weights adversarial.h5
## (0, 0.6044866, u'tench')
## (1, 0.87796897, u'goldfish')
## (2, 0.6075824, u'great_white_shark')
## (3, 0.6018741, u'tiger_shark')
## (4, 0.9041798, u'hammerhead')
## (5, 0.91017926, u'electric_ray')
## (6, 0.95157474, u'stingray')
## (7, 0.8722351, u'cock')
## (8, 0.8244661, u'hen')
## (9, 0.67793137, u'ostrich')


imgs = np.load("imgs.npy")
pred = inception_model.predict(imgs)

for i in range(10):
     print(np.argmax(pred[i]),np.max(pred[i]))

## Gives the following output using trained weights adversarial.h5 
## (0, 0.9732293)
## (1, 0.9910329)
## (2, 0.95925266)
## (3, 0.9316272)
## (4, 0.9861369)
## (5, 0.9819935)
## (6, 0.9761731)
## (7, 0.9852461)
## (8, 0.96967906)
## (9, 0.9955556)

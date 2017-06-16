import ljqpy, os, sys, time, random
import numpy as np
import keras.backend as K
from collections import defaultdict
from keras.datasets import mnist
from keras.utils.generic_utils import Progbar
from keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from keras.preprocessing import image
from PIL import Image
import cv2
time.clock()

np.random.seed(1333)
K.set_image_dim_ordering('tf')

from model import BuildGenerator, BuildDiscriminator

# params
nb_epochs = 500
batch_size = 1
p_gray = 10
p_wgangp = 10
p_beta = 5

channels = 3
imgsize = 256

modeldir = 'data/'
testimgdir = 'images/'

imgdirA     = '/mnt/smb25/ImageNet/erciyuan/train'
testimgdirA = '/mnt/smb25/ImageNet/erciyuan/test'


def Predict(fn):
	img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = np.expand_dims(np.expand_dims(gray, 0), 3)
	gray = (gray - 127.5) / 127.5
	iimg = Input(shape=(gray.shape[1], gray.shape[2], 1))
	modelG = BuildGenerator(iimg)
	modelG.load_weights(os.path.join(modeldir, 'modelG.h5'))
	modelG.compile(optimizer='adam', loss='mse')
	ret = modelG.predict(gray)[0]
	ret = (ret * 127.5 + 127.5).astype(np.uint8)
	Image.fromarray(ret).save(os.path.join(testimgdir, 'pre_'+fn.split('/')[-1]))


def PredictDir(dir):
	for fn in os.listdir(dir):
		fn = os.path.join(dir, fn)
		print(fn)
		Predict(fn)

PredictDir(testimgdirA)

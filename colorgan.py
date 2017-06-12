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
p_beta = 0.2

channels = 3
imgsize = 256

adam_lr = 0.00005
adam_beta_1 = 0.5

imgdirA = '/mnt/smb25/ImageNet/erciyuan/train'
testimgdirA = '/mnt/smb25/ImageNet/erciyuan/test'

modeldir = 'data/'
testimgdir = 'images/'


def imgloss(y_true, y_pred):
	diff = K.abs(y_true - y_pred)
	diff *= [[[[0.29891, 0.58661, 0.11448]]]]
	return K.mean(diff)

def RLinearMerge(x):
	alpha = np.random.uniform(0,1,(batch_size,imgsize,imgsize,channels))
	return x[0] + alpha * (x[1]-x[0])

def GradientsPenalty(x):
	grads = K.gradients( x[1], [x[0]] )[0]
	slopes = K.sqrt(K.sum(K.square(grads), axis=-1))
	gradp = K.mean((slopes - 1.)**2)
	return gradp * p_wgangp

modelG = BuildGenerator(imgsize)
modelD = BuildDiscriminator(imgsize)

#modelG.summary()
#modelD.summary()

try:
	modelG.load_weights(os.path.join(modeldir, 'modelG.h5'))
	modelD.load_weights(os.path.join(modeldir, 'modelD.h5'))	
except Exception as e: 
	print(e)

modelD.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')
modelG.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')

imageReal = Input(shape=(imgsize,imgsize,channels))
imageFake = Input(shape=(imgsize,imgsize,channels))
DReal = modelD(imageReal)
DFake = modelD(imageFake)
inter = merge([imageReal, imageFake], mode=RLinearMerge, output_shape=lambda d:d[0])
gradp = merge([inter, modelD(inter)], mode=GradientsPenalty, output_shape=lambda d:(d[0],1))
combD = Model(inputs=[imageReal, imageFake], outputs=[DReal, DFake, gradp])
combD.compile(optimizer=Adam(adam_lr, adam_beta_1), loss=['mse','mse','mae'])

p_beta_weight = K.variable(p_beta)

def ConvGray(x):
	return x[:,:,:,0:1] * 0.29891 + x[:,:,:,1:2] * 0.58661 + x[:,:,:,2:3] * 0.11448

imageA = Input(shape=(imgsize,imgsize,1))
modelD.trainable = False
fake = modelG(imageA)
disG = modelD(fake)
regray = Lambda(ConvGray, output_shape=lambda d:d[:-1]+(1,))(fake)
combM = Model(inputs=imageA, outputs=[disG, fake, regray])
combM.compile(optimizer=Adam(adam_lr, adam_beta_1), loss=['mse', 'mae', 'mae'], loss_weights=[1,p_beta_weight,p_gray])

def ImgGenerator(imgdir):
	cache = []
	while True:
		imglst = [os.path.join(imgdir, x) for x in os.listdir(imgdir)]
		random.shuffle(imglst)
		for fn in imglst:
			try:
				img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
				w, h = img.shape[1], img.shape[0]
				if w > 1000 or h > 1000: w, h = w // 2, h // 2
				if w > 800 or h > 800: w, h = w // 1.5, h // 1.5
				if w < imgsize or h < imgsize: w, h = w * 1.5, h * 1.5
				img = cv2.resize(img, (int(w), int(h)))
				gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				img, gray = (img - 127.5)/127.5, (gray - 127.5)/127.5
				gray = np.expand_dims(np.expand_dims(gray, 0), 3)
				img = np.expand_dims(img, 0)
				cache.append( (gray, img) )
				if len(cache) > 50:
					for ii in range(2500):
						gimg, cimg = random.choice(cache)
						px, py = random.randint(0, cimg.shape[1]-imgsize), random.randint(0, cimg.shape[2]-imgsize)
						gx, cx = gimg[:,px:px+imgsize,py:py+imgsize,:], cimg[:,px:px+imgsize,py:py+imgsize,:] 
						yield gx, cx
					cache = cache[-3:]
			except: pass

# if the memory is enough for all imgs ...
def ImgGeneratorS(imgdir):
	imglst = [os.path.join(imgdir, x) for x in os.listdir(imgdir)]
	cache = []
	for i, fn in enumerate(imglst):
		if i % 20 == 0: print('%d/%d' % (i, len(imglst)))
		img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
		w, h = img.shape[1], img.shape[0]
		if w > 1000 or h > 1000: w, h = w // 2, h // 2
		if w > 800 or h > 800: w, h = w // 1.5, h // 1.5
		if w < imgsize or h < imgsize: w, h = w * 1.5, h * 1.5
		img = cv2.resize(img, (int(w), int(h)))
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img, gray = (img - 127.5)/127.5, (gray - 127.5)/127.5
		gray = np.expand_dims(np.expand_dims(gray, 0), 3)
		img = np.expand_dims(img, 0)
		for ii in range(50):
			px, py = random.randint(0, img.shape[1]-imgsize), random.randint(0, img.shape[2]-imgsize)
			gx, cx = gray[:,px:px+imgsize,py:py+imgsize,:], img[:,px:px+imgsize,py:py+imgsize,:] 
			cache.append( (gx, cx) )
	ids = list(range(len(cache)))
	while True:
		random.shuffle(ids)
		for ii in ids: 
			yield cache[ii]

gen     = ImgGenerator(imgdirA)
testgen = ImgGeneratorS(testimgdirA)

#nb_batches = len(os.listdir(imgdirA)) // batch_size
nb_batches = 2999
ones  = np.ones(  (batch_size, imgsize//8, imgsize//8, 1) )
zeros = np.zeros( (batch_size, imgsize//8, imgsize//8, 1) )

record = []
for epoch in range(nb_epochs):
	print('Epoch %d of %d' % (epoch+1, nb_epochs))
	progress_bar = Progbar(target=nb_batches)
	
	for index in range(nb_batches):
		for _ in range(5):
			gimg, cimg = next(gen)
			generate = modelG.predict_on_batch(gimg)
			record.append(generate)
			if len(record) > 100: record = record[-50:]
			lossD = combD.train_on_batch([cimg, random.choice(record)], [ones, zeros, np.zeros((batch_size,1))])[0]

		for _ in range(1):
			gimg, cimg = next(gen)
			__, lossG, lossR, lossGray = combM.train_on_batch(gimg, [ones, cimg, gimg])
			
		progress_bar.update(index+1, values=[('D',lossD),('G',lossG),('R',lossR),('Gray',lossGray)])

	p_beta = np.clip(p_beta * 0.95, 0.01, 10)
	K.set_value(p_beta_weight, p_beta)

	print('Testing for epoch {}:'.format(epoch + 1))
	print('p_beta=%.3f' % p_beta)

	modelG.save_weights( os.path.join(modeldir, 'modelG.h5' ), True)
	modelD.save_weights( os.path.join(modeldir, 'modelD.h5' ), True)
	
	tests = [next(testgen) for x in range(4)]
	graytest, colortest = [x for x,y in tests], [y for x,y in tests]
	tGray = np.concatenate( graytest, axis=0 ) 
	tCol = np.concatenate( colortest, axis=0 ) 
	tGen = modelG.predict(tGray, batch_size=1)
	tGray = np.repeat(tGray,3)
	tGray = tGray.reshape(-1,imgsize,channels)
	tCol = tCol.reshape(-1,imgsize,channels)
	tGen = tGen.reshape(-1,imgsize,channels)
	img = np.concatenate([tGray, tGen, tCol], axis=1)
	img = (img * 127.5 + 127.5).astype(np.uint8)  
	Image.fromarray(img).save(os.path.join(testimgdir, 'plot_epoch_{0:03d}_generated.png'.format(epoch)))

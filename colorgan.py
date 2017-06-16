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
p_wgangp = 10
p_gray = 10
p_beta = 10

channels = 3
imgsize = 256

adam_lr = 0.00005
adam_beta_1 = 0.5

imgdirA     = '/mnt/smb25/ImageNet/erciyuan/train'
testimgdirA = '/mnt/smb25/ImageNet/erciyuan/test'

modeldir   = 'data/'
testimgdir = 'images/'


def imgloss(y_true, y_pred):
	diff = K.abs(y_true - y_pred)
	diff *= [[[[0.29891, 0.58661, 0.11448]]]]
	return K.mean(diff)

def RLinearMerge(x):
	alpha = np.random.uniform(0,1,(batch_size,1))
	return x[0] + alpha * (x[1]-x[0])

def GradientsPenalty(x):
	grads = K.gradients( x[1], [x[0]] )[0]
	slopes = K.sqrt(K.sum(K.square(grads), axis=-1))
	gradp = K.mean((slopes - 1.)**2)
	return gradp

def wloss(y_true, y_pred):
	return - K.mean(y_pred)


img = Input(shape=(imgsize,imgsize,1))
modelG = BuildGenerator(img)
img = Input(shape=(imgsize,imgsize,3))
modelD = BuildDiscriminator(img)

modelG.summary()
modelD.summary()

try:
	modelG.load_weights(os.path.join(modeldir, 'modelG.h5'))
	modelD.load_weights(os.path.join(modeldir, 'modelD.h5'))	
except Exception as e: 
	print(e); print('new model')

modelD.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')
modelG.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')

imageReal = Input(shape=(imgsize,imgsize,channels))
imageFake = Input(shape=(imgsize,imgsize,channels))
DReal = modelD(imageReal)
DFake = modelD(imageFake)
inter = merge([imageReal, imageFake], mode=RLinearMerge, output_shape=lambda d:d[0])
gradp = merge([inter, modelD(inter)], mode=GradientsPenalty, output_shape=lambda d:(d[0],1))
combD = Model(inputs=[imageReal, imageFake], outputs=[DReal, DFake, gradp])
combD.compile(optimizer=Adam(adam_lr, adam_beta_1), loss=['mse', 'mse', 'mae'], loss_weights=[1, 1, p_wgangp])

p_beta_weight = K.variable(p_beta)

def ConvGray(x):
	return x[:,:,:,0:1] * 0.29891 + x[:,:,:,1:2] * 0.58661 + x[:,:,:,2:3] * 0.11448

imageA = Input(shape=(imgsize,imgsize,1))
modelD.trainable = False
fake = modelG(imageA)
disG = modelD(fake)
regray = Lambda(ConvGray, output_shape=lambda d:d[:-1]+(1,))(fake)
combM = Model(inputs=imageA, outputs=[disG, fake, regray])
combM.compile(optimizer=Adam(adam_lr, adam_beta_1), loss=['mse', 'mae', 'mae'], loss_weights=[1,p_beta_weight,p_beta_weight])

def ResizeRatio(w, h):
	short, long = min(w, h), max(w, h)
	if short <= imgsize: obj = random.randint(imgsize+1, imgsize+10)
	elif long >= short * 2.5: obj = random.randint(imgsize+1, imgsize+10)
	elif imgsize < short < imgsize * 2: obj = random.randint(imgsize+1, short)
	else: obj = random.randint(imgsize+1, int(imgsize * 2))
	ratio = obj / short
	if ratio < 0.3: ratio = 0.3
	return ratio

def ImgGenerator(imgdir):
	cache = []
	while True:
		imglst = [os.path.join(imgdir, x) for x in os.listdir(imgdir)]
		random.shuffle(imglst)
		for fn in imglst:
			try:
				img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
				w, h = img.shape[1], img.shape[0]
				ratio = ResizeRatio(w, h)
				img = cv2.resize(img, (int(w*ratio+.5), int(h*ratio+.5)))
				gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				img, gray = (img - 127.5)/127.5, (gray - 127.5)/127.5
				gray = np.expand_dims(np.expand_dims(gray, 0), 3)
				img = np.expand_dims(img, 0)
				cache.append( (gray, img) )
				if len(cache) > 400:
					for ii in range(7000):
						gimg, cimg = random.choice(cache)
						px, py = random.randint(0, cimg.shape[1]-imgsize), random.randint(0, cimg.shape[2]-imgsize)
						gx, cx = gimg[:,px:px+imgsize,py:py+imgsize,:], cimg[:,px:px+imgsize,py:py+imgsize,:] 
						yield gx, cx
					cache = cache[-3:]
			except Exception as e: 
				print('bad file:', fn, e)

# if the memory is enough for all imgs ...
def ImgGeneratorS(imgdir):
	imglst = [os.path.join(imgdir, x) for x in os.listdir(imgdir)]
	cache = []
	for i, fn in enumerate(imglst+imglst):
		if i % 20 == 0: print('%d/%d' % (i, len(imglst)))
		img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
		w, h = img.shape[1], img.shape[0]
		ratio = ResizeRatio(w, h)
		img = cv2.resize(img, (int(w*ratio+.5), int(h*ratio+.5)))
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img, gray = (img - 127.5)/127.5, (gray - 127.5)/127.5
		gray = np.expand_dims(np.expand_dims(gray, 0), 3)
		img = np.expand_dims(img, 0)
		for ii in range(10):
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
ones  = np.ones(  (batch_size, imgsize//16, imgsize//16, 1) )
zeros = np.zeros( (batch_size, imgsize//16, imgsize//16, 1) )


record = []
objD = [np.concatenate([ones]*batch_size,0), np.concatenate([zeros]*batch_size,0), np.zeros((batch_size,1))]
for epoch in range(nb_epochs):
	print('Epoch %d of %d' % (epoch+1, nb_epochs))
	progress_bar = Progbar(target=nb_batches)
	
	for index in range(nb_batches):
		D_iter = 2
		lossD = 0; 
		for _ in range(D_iter):
			gimg, cimg = next(gen)
			generate = modelG.predict_on_batch(gimg)
			record.append(generate)
			if len(record) > 50: record = record[-25:]
			lossD += combD.train_on_batch([cimg, random.choice(record)], objD)[0] / D_iter

		for _ in range(1):
			gimg, cimg = next(gen)
			__, lossG, lossR, lossGray = combM.train_on_batch(gimg, [ones, cimg, gimg])
			
		progress_bar.update(index+1, values=[('D',lossD),('G',lossG),('R',lossR),('Gray',lossGray)])
	
	p_beta = np.clip(p_beta * 0.9, 0.1, 10)
	K.set_value(p_beta_weight, p_beta)

	print('Testing for epoch {}:'.format(epoch + 1))
	print('p_beta=%.3f' % p_beta)
	
	if epoch % 5 == 4:
		lr =  K.get_value(combD.optimizer.lr) * 0.9
		print('lr=%.8f' % lr)
		K.set_value( combD.optimizer.lr, lr )
		K.set_value( combM.optimizer.lr, lr )

	modelG.save_weights( os.path.join(modeldir, 'modelG.h5' ), True)
	modelD.save_weights( os.path.join(modeldir, 'modelD.h5' ), True)
	
	numtest = 10
	tests = [next(testgen) for x in range(numtest)]
	graytest, colortest = [x for x,y in tests], [y for x,y in tests]
	tGray = np.concatenate( graytest, axis=0 ) 
	tCol = np.concatenate( colortest, axis=0 ) 
	tGen = modelG.predict(tGray, batch_size=1)
	obj = [np.concatenate([ones]*numtest,0), np.concatenate([zeros]*numtest,0), np.zeros((numtest,1))]
	_, lossReal, lossFake, lossGP = combD.evaluate([tCol, tGen], obj, batch_size=1, verbose=0)
	_, lossG, lossR, lossGray = combM.evaluate(tGray, [obj[0], tCol, tGray], batch_size=1, verbose=0)
	print('lossReal=%.4f, lossFake=%.4f, lossGP=%.4f' % (lossReal, lossFake, lossGP))
	print('lossG=%.4f, lossR=%.4f, lossGray=%.4f' % (lossG, lossR, lossGray))
	tGray = np.repeat(tGray,3)
	tGray = tGray.reshape(-1,imgsize,channels)
	tCol = tCol.reshape(-1,imgsize,channels)
	tGen = tGen.reshape(-1,imgsize,channels)
	img = np.concatenate([tGray, tGen, tCol], axis=1)
	img = (img * 127.5 + 127.5).astype(np.uint8)  
	Image.fromarray(img).save(os.path.join(testimgdir, 'plot_epoch_{0:03d}_generated.png'.format(epoch)))
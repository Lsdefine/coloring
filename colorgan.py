import ljqpy, os, sys, time, random
import numpy as np
import keras.backend as K
import tensorflow as tf
from collections import defaultdict
from keras.datasets import mnist
from keras.utils.generic_utils import Progbar
from keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from keras.preprocessing import image
from keras.utils import data_utils
from PIL import Image
import cv2
time.clock()

np.random.seed(1335)
K.set_image_dim_ordering('tf')
from model import BuildGenerator, BuildDiscriminator

# params
nb_epochs = 100
batch_size = 1
p_wgangp = 10
p_recon = 10
p_gray = 10
p_diff = 1

channels = 3
imgsize = 256

adam_lr = 0.000002
adam_beta_1 = 0.5

imgdirA     = '/mnt/smb25/ImageNet/erciyuan/train'
testimgdirA = '/mnt/smb25/ImageNet/erciyuan/test'

modeldir   = 'data/'
testimgdir = 'images/'

colorweight = np.array([[[[0.29891, 0.58661, 0.11448]]]]) * 3
def imgloss(y_true, y_pred):
	diff = K.abs(y_true - y_pred)
	diff *= colorweight
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
	return - K.clip(K.mean(y_pred), -50, 50)

img = Input(shape=(imgsize,imgsize,1))
modelG = BuildGenerator(img)
img = Input(shape=(imgsize,imgsize,3))
modelD = BuildDiscriminator(img)

try:
	modelG.load_weights(os.path.join(modeldir, 'modelG.h5'))
	modelD.load_weights(os.path.join(modeldir, 'modelD.h5'))	
except Exception as e: 
	print(e); print('new model')
	modelG.summary()
	modelD.summary()

modelD.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')
modelG.compile(optimizer=Adam(adam_lr, adam_beta_1), loss='mse')

imageReal = Input(shape=(imgsize,imgsize,channels))
imageFake = Input(shape=(imgsize,imgsize,channels))
DReal = modelD(imageReal)
DFake = modelD(imageFake)
#inter = merge([imageReal, imageFake], mode=RLinearMerge, output_shape=lambda d:d[0])
#gradp = merge([inter, modelD(inter)], mode=GradientsPenalty, output_shape=lambda d:(d[0],1))
combD = Model(inputs=[imageReal, imageFake], outputs=[DReal, DFake])
combD.compile(optimizer=Adam(adam_lr, adam_beta_1), loss=['mse', 'mse'], loss_weights=[1, 1])

p_recon_weight = K.variable(p_recon)

def ConvGray(x):
	return x[:,:,:,0:1] * 0.29891 + x[:,:,:,1:2] * 0.58661 + x[:,:,:,2:3] * 0.11448

def GetDiffX(x): return K.max(K.abs(x[:,1:,:,:] - x[:,:-1,:,:]), -1)
def GetDiffY(x): return K.max(K.abs(x[:,:,1:,:] - x[:,:,:-1,:]), -1)

imageA = Input(shape=(imgsize,imgsize,1))
modelD.trainable = False
fake = modelG(imageA)
disG = modelD(fake)
regray = Lambda(ConvGray, output_shape=lambda d:d[:-1]+(1,))(fake)

difffakeX = Lambda(GetDiffX, output_shape=lambda d:(d[0], d[1]-1, d[2], 1))(fake)
diffgrayX = Lambda(GetDiffX, output_shape=lambda d:(d[0], d[1]-1, d[2], 1))(imageA)
difffakeY = Lambda(GetDiffY, output_shape=lambda d:(d[0], d[1], d[2]-1, 1))(fake)
diffgrayY = Lambda(GetDiffY, output_shape=lambda d:(d[0], d[1], d[2]-1, 1))(imageA)
diffX = merge([difffakeX, diffgrayX], mode=lambda d:K.mean(K.relu(d[0]-d[1])), output_shape=lambda d:(d[0][0], 1))
diffY = merge([difffakeY, diffgrayY], mode=lambda d:K.mean(K.relu(d[0]-d[1])), output_shape=lambda d:(d[0][0], 1))
diff = add([diffX, diffY])
combM = Model(inputs=imageA, outputs=[disG, fake, regray, diff])
combM.compile(optimizer=Adam(adam_lr, adam_beta_1), loss=['mse', imgloss, 'mae', 'mae'], loss_weights=[1,p_recon_weight,p_gray,p_diff])

def ResizeRatio(w, h):
	short, long = min(w, h), max(w, h)
	mxx = short
	if short > 1500: mxx = short // 3
	elif short > 1000: mxx = short // 2
	if short <= imgsize: mxx = imgsize + 10
	if long >= short * 2.5: mxx = imgsize + 10
	obj = random.randint(imgsize+5, mxx)
	ratio = obj / short
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
				if random.random() < 0.4: img = img[:,::-1,:]
				gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				img, gray = img / 127.5 - 1, gray / 127.5 - 1
				gray = np.expand_dims(np.expand_dims(gray, 0), 3)
				img = np.expand_dims(img, 0)
				cache.append( (gray, img) )
				if len(cache) > 300:
					for ii in range(3000):
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
		if random.random() < 0.4: img = img[:,::-1,:]
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img, gray = img / 127.5 - 1, gray / 127.5 - 1
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

import queue, threading
dqueue = queue.Queue()

def QueuePut():
	while True:
		if dqueue.qsize() < 5000: dqueue.put(next(gen))
		else: time.sleep(0.5)
threading._start_new_thread(QueuePut, tuple())
def QueueGet():
	while True:
		if not dqueue.empty():
			ret = dqueue.get()
			yield ret
		else: time.sleep(0.5)
syncgen = QueueGet()

#nb_batches = len(os.listdir(imgdirA)) // batch_size
nb_batches = 999
ones  = np.ones(  (batch_size, imgsize//16, imgsize//16, 1) )
zeros = np.zeros( (batch_size, imgsize//16, imgsize//16, 1) )

record = []
objD = [np.concatenate([ones]*batch_size,0), np.concatenate([zeros]*batch_size,0), np.zeros((batch_size,1))][:2]
for epoch in range(nb_epochs):
	print('Epoch %d of %d' % (epoch, nb_epochs))
	
	D_iter = 1;  G_iter = 1;
	if epoch > 0: D_iter = 7; G_iter = 3
	progress_bar = Progbar(target=nb_batches*(G_iter+D_iter))
	iter = 1
	for index in range(nb_batches):
		for _ in range(D_iter):
			gimg, cimg = next(syncgen)
			generate = modelG.predict_on_batch(gimg)
			record.append(generate)
			if len(record) > 100: record = record[-50:]
			lossD = combD.train_on_batch([cimg, random.choice(record)], objD)[0]
			progress_bar.update(iter, values=[('D',lossD)])
			iter += 1

		for _ in range(G_iter):
			gimg, cimg = next(syncgen)
			__, lossG, lossR, lossGray, lossDiff = combM.train_on_batch(gimg, [ones, cimg, gimg, np.zeros((batch_size, 1))])
			progress_bar.update(iter, values=[('G',lossG),('R',lossR),('Gray',lossGray),('Diff',lossDiff)])
			iter += 1
	
	#p_recon = np.clip(p_recon * 0.9, 1, 10)
	#K.set_value(p_recon_weight, p_recon)

	print('Testing for epoch {}:'.format(epoch))
	print('p_recon=%.3f' % p_recon)
	
	if epoch % 4 == 3:
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
	obj = [np.concatenate([ones]*numtest,0), np.concatenate([zeros]*numtest,0), np.zeros((numtest,1))][:2]
	dloss = combD.evaluate([tCol, tGen], obj, batch_size=1, verbose=0)
	mloss = combM.evaluate(tGray, [obj[0], tCol, tGray, np.zeros((numtest, 1))], batch_size=1, verbose=0)
	dloss += (0,)
	print('lossReal=%.4f, lossFake=%.4f, lossGP=%.4f' % (dloss[1], dloss[2], dloss[3]))
	print('lossG=%.4f, lossR=%.4f, lossGray=%.4f, lossDiff=%.4f' % (mloss[1], mloss[2], mloss[3], mloss[4]))
	tGray = np.repeat(tGray,3)
	tGray = tGray.reshape(-1,imgsize,channels)
	tCol = tCol.reshape(-1,imgsize,channels)
	tGen = tGen.reshape(-1,imgsize,channels)
	img = np.concatenate([tGray, tGen, tCol], axis=1)
	img = (img * 127.5 + 127.5).astype(np.uint8)  
	Image.fromarray(img).save(os.path.join(testimgdir, 'plot_epoch_{0:03d}_generated.png'.format(epoch)))
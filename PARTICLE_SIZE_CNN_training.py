# -*- coding: utf-8 -*-
"""
Created on %(date)s

@Code author: Hamid Ghanbari
Email : hamid.ghanbari.1@ulaval.ca
Paper : Convolutional neural networks for mapping of lake sediment core particle size using hyperspectral imaging
authors: H. Ghanbari, D. Antoniades
"""
from keras.regularizers import l2
from scipy.signal import savgol_filter
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import dtcwt
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from sklearn.ensemble import RandomForestRegressor
import pickle
import scipy
from scipy.signal import find_peaks
from keras.layers import Lambda, Input, Dense, Activation, ZeroPadding1D, multiply,BatchNormalization, Flatten, Conv1D, InputLayer, Reshape, UpSampling1D, Concatenate
from keras.layers import LeakyReLU, MaxPooling1D, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D, Add, Multiply
from keras.models import Model, Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras import regularizers
from sklearn.decomposition import PCA
from keras.losses import mse
from keras.constraints import unit_norm
from sklearn.manifold import MDS
from keras import backend as K
import time
def mean_relative_error(y_true , y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true)) *100



def CNNmodel1(input_shape = (64,1), classes = 128):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
 
    
    X_input = Input(input_shape)

    X1 = Conv1D(16, 3, strides = 1, padding='same', name = 'conv0')(X_input)
    X1 = BatchNormalization(axis = 1, name = 'bn0')(X1)
    X1 = LeakyReLU()(X1)
    X3 = Conv1D(32, 3, strides = 1, padding='same', name = 'conv2')(X1)
    X3 = BatchNormalization(axis = 1, name = 'bn2')(X3)
    X3 = LeakyReLU()(X3)
    X3 = MaxPooling1D(pool_size=2, name='max_pool2')(X3)
    X4 = Conv1D(64, 3, strides = 1, padding='same', name = 'conv3')(X3)
    X4 = BatchNormalization(axis = 1, name = 'bn3')(X4)
    X4 = LeakyReLU()(X4)
    X4 = MaxPooling1D(pool_size=4, name='max_pool3')(X4)
    X5 = Conv1D(64, 5, strides = 1, padding='same', name = 'conv4')(X4)
    X5 = BatchNormalization(axis = 1, name = 'bn4')(X5)
    X5 = Activation('relu')(X5)
    X5 = MaxPooling1D(pool_size=4, name='max_pool4')(X5)
    X_shortcut1 =  cbam_block(X1, ratio=8)
    X_shortcut1 = Reshape((16,64))(X_shortcut1)
    X_shortcut1 = BatchNormalization(axis = 1)(X_shortcut1)
    X_shortcut1 = LeakyReLU()(X_shortcut1)
    X_shortcut1 = MaxPooling1D(pool_size=8)(X_shortcut1)



    X = Concatenate(axis=1)([Flatten()(X_shortcut1) , Flatten()(X5)])
    X = Dropout(0.4)(X)
    X = Dense(64, activation='relu', name='fc2', kernel_regularizer=l2(0.001))(X)#
    X = Dense(1, activation='relu', name='fc4')(X)
    model = Model(inputs = X_input, outputs = X, name='CNNmodel')

    return model   

def build_autoencoder(img_shape, code_size):
    # The encoder
    # encoder = Sequential()
    original_dim = img_shape[0]
    encoder_input  = Input(shape=(original_dim, ), name='encoder_input1')
    inputs_x1 = Reshape((original_dim, 1), input_shape=(original_dim,))(encoder_input)
    E1 = Conv1D(16, 5, strides = 1, padding='same', name= 'encode-cnn')(inputs_x1)
    E1 = LeakyReLU()(E1)
    E1 = MaxPooling1D(pool_size= 4)(E1)
    
    E2 = Conv1D(32, 5, strides = 1, padding='same', name= 'encode-cnn1')(E1)
    E2 = LeakyReLU()(E2)
    E3 = MaxPooling1D(pool_size= 4)(E2)
    
    E4 = Conv1D(64, 5, strides = 1, padding='same', name= 'encode-cnn2')(E3)
    E4 = LeakyReLU()(E4)
    E5 = MaxPooling1D(pool_size= 4)(E4)    
    E6 = Conv1D(64, 15, strides = 15, padding='same', name= 'encode-cnn3')(E5)#, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)
    E6 = LeakyReLU()(E6)

    
    E8 = Flatten()(E6)

    encoder = Model(encoder_input, E8, name='encoder')
    
    decoder_input  = Input(shape=64, name='decoder_input1')
    D3 = Reshape(( 64,1), input_shape=(64,))(decoder_input)
    D4 = Conv1D(15, 3, strides = 1, padding='same', name= 'decode-cnn2')(D3)
    D4 = LeakyReLU()(D4)
    D6 = Conv1D(60, 3, strides = 1, padding='same', name= 'decode-cnn3')(D4)
    D6 = LeakyReLU()(D6)
    D7 = MaxPooling1D(pool_size= 2)(D6)
    D8 = Conv1D(240, 3, strides = 1, padding='same', name= 'decode-cnn4')(D7)
    D8 = LeakyReLU()(D8)
    D9 = MaxPooling1D(pool_size= 2)(D8) 
    D10 = Conv1D(960, 3, strides = 1, padding='same', name= 'decode-cnn5', activation = 'linear')(D9) 
    D10 = MaxPooling1D(pool_size= 16)(D10) 
    D10 = Flatten()(D10)
    decoder = Model(decoder_input, D10, name='decoder')

    return encoder, decoder


def cbam_block(cbam_feature, ratio=8):

    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel = input_feature.shape[-1]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    
    avg_pool = GlobalAveragePooling1D()(input_feature)    
    avg_pool = Reshape((1,channel))(avg_pool)   
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=2)([avg_pool, max_pool])
    cbam_feature = Conv1D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)    

    return multiply([input_feature, cbam_feature])



start = time.time()

names =['BEC' , 'STA','WIL','TRU', 'PSF','GL','SL','JOS','EL']# 
resolutions = [13 , 13, 13, 13,13, 13,13,13,13,13,13,13,13 ]
wl1 = np.linspace(395.82 , 1001.49 , 776)
wl2 = np.linspace(969.29 , 2577.33 , 288)
wl = np.concatenate((wl1 , wl2))
wl = wl[list(np.concatenate((np.array(range(75,765)),np.array(range(782,1052)))))]
Ref= []
sample = []
counts = 0
for name in names:
    print(name)    
    maskpath = rf'C:\hamid\Projects\grain size in sediment core_AnalySize\{name}\VSWIR_mask.hdr'
    filename =rf'C:\hamid\Projects\grain size in sediment core_AnalySize\{name}\VSWIR.hdr'
    mask = envi.open(maskpath)
    mask = mask.load()
    mask[np.where(mask==1)] = np.nan
    mask[np.where(mask==0)] = 1
    data = envi.open(filename)
    data = data[:,:,list(np.concatenate((np.array(range(75,765)),np.array(range(782,1052)))))]
    [r,c,b] = data.shape
    mask = mask.reshape(r,c,1)
    data = data *mask
    mskBand = 50 
    Wband = ((data[:,:,mskBand] + data[:,:,mskBand+1])/2).reshape(data.shape[0:2])
    Msk = np.ones(np.shape(Wband))
    ranges = np.linspace(0, Wband.shape[0],Wband.shape[0], dtype = 'int')    
    for item in range(Wband.shape[0]-1):
        a  = np.ones((ranges[item+1]-ranges[item], Wband.shape[1]))
        Mean = np.nanmedian(Wband[ranges[item]:ranges[item+1] , :])
        STD = np.nanstd(Wband[ranges[item]:ranges[item+1] , :])
        Id1 = np.where(Wband[ranges[item]:ranges[item+1], :] < Mean- 3*STD)
        a[Id1] = np.NaN
        Id2 = np.where(Wband[ranges[item]:ranges[item+1], :] > Mean+ 3*STD)
        a[Id2] = np.NaN
        Msk[ranges[item]:ranges[item+1], :] = a
        
    Msk = Msk.reshape(r,c,1)#*mask
    data = data *Msk

    mskBand = 850 
    Wband = ((data[:,:,mskBand] + data[:,:,mskBand+1])/2).reshape(data.shape[0:2])
    Msk = np.ones(np.shape(Wband))
    ranges = np.linspace(0, Wband.shape[0],Wband.shape[0], dtype = 'int')    
    for item in range(Wband.shape[0]-1):
        a  = np.ones((ranges[item+1]-ranges[item], Wband.shape[1]))
        Mean = np.nanmedian(Wband[ranges[item]:ranges[item+1] , :])
        STD = np.nanstd(Wband[ranges[item]:ranges[item+1] , :])
        Id1 = np.where(Wband[ranges[item]:ranges[item+1], :] < Mean- 3*STD)
        a[Id1] = np.NaN
        Id2 = np.where(Wband[ranges[item]:ranges[item+1], :] > Mean+ 3*STD)
        a[Id2] = np.NaN
        Msk[ranges[item]:ranges[item+1], :] = a
        
    Msk = Msk.reshape(r,c,1)#*mask
    data = data *Msk
    
    Grain_ref = pd.read_excel(r'\Grain size data Reference.xlsx', sheet_name= f'{name}').values
    length = Grain_ref[-1,0]+.5
    Grain_ref = Grain_ref[:-1,:]
    interval = 1
    ref = Grain_ref[:,1]
    ranges = np.zeros((ref.shape[0]), dtype = 'int')    
    y = np.linspace(0, length , data.shape[0])
    for item in range(ref.shape[0]):
        ranges[item] = np.where(np.abs(y-Grain_ref[item, 0]) == np.min(np.abs(y-Grain_ref[item, 0])))[0]
    res = resolutions[counts]
    
    for counters in range(np.size(ref)):
        ranges_x = np.linspace(ranges[counters]-res , ranges[counters]+res,2, dtype= 'int')# int(res) 
        ranges_y = np.linspace(0 ,data.shape[1],2, dtype = 'int')
        for i in range(len(ranges_x)-1):
            for j in range(len(ranges_y)-1):
                box = data[ranges_x[i] :ranges_x[i+1],ranges_y[j] :ranges_y[j+1],:]
                if ~(np.isnan(box).all()):
                    mean_spectrum = np.nanmean(np.nanmean(box,0).T , 1).squeeze()
                    dif = box - mean_spectrum
                    dif_sum = np.sum(np.power(dif, 2) , axis = 2)
                    r1 , c1 = np.where(dif_sum == np.nanmin(dif_sum))
                    if ~(np.isnan(box[r1, c1 , :]).all()):
                        sample.append(box[r1[0], c1[0] , :].squeeze())
                        Ref.append(ref[counters])

    counts +=1
    peaks1 = np.ones(len(sample))
    peaks2 = np.ones(len(sample))
    for k in range(len(sample)):
        peaks, _ = find_peaks(sample[k][:653], height=0 , width= 2, threshold=0.000005)
        peaks1[k] = len(peaks)
        peaks, _ = find_peaks(sample[k][660:], height=0 , width= 2, threshold=0.0001)
        peaks2[k] = len(peaks)
    IDs1 = np.where(peaks1 < np.median(peaks1) + 3 * np.std(peaks1))[0]
    IDs2 = np.where(peaks2 < np.mean(peaks2) + 3 * np.std(peaks2))[0]
    IDs = np.intersect1d(IDs1 , IDs2)
    sample = list(np.array(sample)[IDs])
    Ref = list(np.array(Ref)[IDs])
sample = np.array(sample)
ref = np.array(Ref)
IDs = np.where(~np.isnan(sample.sum(1)))
sample = sample[IDs]
ref = ref[IDs]
STDS = np.max(sample, axis= 1) - np.min(sample, axis= 1)
IDs = np.where(STDS < np.mean(STDS) + 2.5 * np.std(STDS))[0]
sample = sample[IDs]
ref = ref[IDs]
sample[:,690:] = sample[:,690:] - (sample[:,690] -sample[:,689]   ).reshape(-1 ,1)
cosine_similarity = np.zeros(len(sample))
a = np.mean(sample , axis = 0)
for j in range(len(sample)):
    cosine_similarity[j] = 1 - scipy.spatial.distance.cosine(a, sample[j, :])
IDs = np.where(cosine_similarity > 0.95)[0]
sample = sample[IDs]
ref = ref[IDs]
sample = savgol_filter(sample, 7, 2)
a = sample[:,:689]
b = scipy.signal.medfilt(a , 15)
diff = np.mean(np.abs(a-b), 1)
stda = np.std(diff)
meana = np.mean(diff)
IDs1 = np.where(diff < meana + stda)[0]
IDs2 = np.where(diff > meana - stda)[0]
IDs = np.intersect1d(IDs1 , IDs2)
sample = sample[IDs]
ref = ref[IDs]
sample = (sample - np.mean(sample, 1, keepdims= True)) / (np.std(sample, 1, keepdims= True)) 

if sample.shape[0] %2:
    sample = sample[:-1 , :] 

n_components = np.min(sample.shape)
pca = PCA(n_components=n_components, svd_solver='full')
pca.fit(sample)
samplePC = pca.transform(sample)
samplePC1 = samplePC[:, 5:]
transform = dtcwt.Transform1d()
dwt = transform.forward(samplePC1, nlevels=5)
highpasses = dwt.highpasses

for k in range(5):
    d = highpasses[k]
    
    for j in range(0,d.shape[1]):
        if j == 0:
            s = np.abs(d[:,j])**2
        elif j == d.shape[1]-1:
            s = np.abs(d[:,j])**2
        else: 
            s = (np.abs(d[:,j-1])**2+np.abs(d[:,j])**2+np.abs(d[:,j+1])**2)/3
        sigma = np.median(np.abs(np.abs(highpasses[0][:,j]) - np.mean(np.abs(highpasses[0][:,j])) ))/0.6745
        thr = (2 * sigma**2 * np.log(d.shape[0]))**0.5
        highpasses[k][:,j] = highpasses[k][:,j] * np.maximum(np.zeros(d.shape[0],), (1 -thr**2 /s  ))
dwt.highpasses = highpasses
samplePC2 = transform.inverse(dwt)
samplePC[:, 5:] = samplePC2
sample = pca.inverse_transform(samplePC)

a = [i for i in range(len(sample))]
a = shuffle(a)

train = sample[a,:]
target = ref[a]
# target = np.log(target)
X = target.reshape(-1,1)
from sklearn.mixture import GaussianMixture
N = np.arange(1, 11)
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i] = GaussianMixture(N[i]).fit(X)


M_best = models[2]

x = np.linspace(0, 300, 1000)
logprob = M_best.score_samples(x.reshape(-1, 1))
responsibilities = M_best.predict_proba(x.reshape(-1, 1))
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]



weights = np.log(1 / np.exp(M_best.score_samples(target.reshape(-1,1)))) #
    
 
modelnum = 0
#######Autoencode

print('building autoencoder')
encoder, decoder = build_autoencoder((960,1), 64)
inp = Input((960,1))
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='Adam', loss='mse')#),loss_weights=weights
history = autoencoder.fit(x= train, y=train, epochs=200, validation_split= 0.2 , batch_size = 64, shuffle = True)

code = encoder.predict([train.reshape(train.shape[0], 960, 1) ]).squeeze()
pathname_autoencode = '\CAE_encoder_model.h5'
encoder.save(pathname_autoencode)
   
a = np.histogram(target, 20)
b =  np.cumsum(a[0])/np.sum(a[0])
b = np.append([0] , b)
Dist = np.linspace(b.min() , 1 , 10 )
f1 = scipy.interpolate.interp1d( b,a[1] )
intervals = f1(Dist)
ID_train = []
ID_test = []
for iter in range(9):
    IDs1 = np.where(target <= intervals[iter+1])[0]
    IDs2 = np.where(target > intervals[iter])[0]
    IDs = np.intersect1d(IDs1 , IDs2)
    if len(IDs):
      ID_train1 , ID_test1 = train_test_split(IDs, test_size=0.2, random_state=42)
      ID_train.append(ID_train1)
      ID_test.append(ID_test1)
        # ID_train.append(np.random.choice(IDs, int(len(target)/100)))
ID_train = np.concatenate(ID_train)
ID_test = np.concatenate(ID_test)
X_test = code[ID_test,:]
Y_test = target[ID_test]
X_train = code[ID_train,:]
Y_train = target[ID_train]
X_train = code#[target<thresh,:]
Y_train = target#[target<thresh]

##########CNN
print('building CNN regression')
CNN_model = 1
if CNN_model ==0:
    model = RandomForestRegressor(n_estimators= 500)
    model.fit(X_train, Y_train)
else:
    
    model = CNNmodel1(input_shape = (64,1), classes = 64 )
    model.compile(optimizer = 'Adam',  loss = 'mse')#,loss_weights=weights
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
history = model.fit(x = X_train , y = Y_train  ,validation_data= (X_test , Y_test)  ,epochs = 200, batch_size = 64, shuffle = True, callbacks=[callback])# 
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
filename_CNN = '\MPS_CNN_model1.h5'
model.save(filename_CNN)
print("Saved model to disk")
 

end = time.time()
print("Elappsed Time" , end - start , "seconds")

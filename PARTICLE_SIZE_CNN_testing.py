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
from sklearn.metrics import mean_squared_error, r2_score
def mean_relative_error(y_true , y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true)) *100



def CNNmodel1(input_shape = (64,1), classes = 128):

 
    
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
res = 13
mask_input = 1
names =['BEC' , 'STA','WIL','TRU', 'PSF','GL','SL','JOS','EL']# 
name = 'BEC'#Dataset name you wish to predict the particle size
filename =rf'\{name}\VSWIR.HDR'    
data = envi.open(filename)
data = data[:,:,list(np.concatenate((np.array(range(75,765)),np.array(range(782,1052)))))]
[r,c,b] = data.shape    

if mask_input==1:
    maskpath = rf'C:\hamid\Projects\grain size in sediment core_AnalySize\{name}-1\VSWIR_mask.hdr'
    mask = envi.open(maskpath)
    mask = mask.load()
    mask[np.where(mask==1)] = np.nan
    mask[np.where(mask==0)] = 1
    mask = mask.squeeze()
    mskBand = 50 
    Wband = ((data[:,:,mskBand] + data[:,:,mskBand+1])/2).reshape(data.shape[0:2])
    msk = np.ones(np.shape(Wband))
    ranges = np.linspace(0, Wband.shape[0],Wband.shape[0], dtype = 'int')    
    for item in range(Wband.shape[0]-1):
        a  = np.ones((ranges[item+1]-ranges[item], Wband.shape[1]))
        Mean = np.nanmedian(Wband[ranges[item]:ranges[item+1] , :])
        STD = np.nanstd(Wband[ranges[item]:ranges[item+1] , :])
        Id1 = np.where(Wband[ranges[item]:ranges[item+1], :] < Mean- 3*STD)
        a[Id1] = np.NaN
        Id2 = np.where(Wband[ranges[item]:ranges[item+1], :] > Mean+ 3*STD)
        a[Id2] = np.NaN
        msk[ranges[item]:ranges[item+1], :] = a
    mskBand = 850 
    Wband = ((data[:,:,mskBand] + data[:,:,mskBand+1])/2).reshape(data.shape[0:2])
    msk1 = np.ones(np.shape(Wband))
    ranges = np.linspace(0, Wband.shape[0],Wband.shape[0], dtype = 'int')    
    for item in range(Wband.shape[0]-1):
        a  = np.ones((ranges[item+1]-ranges[item], Wband.shape[1]))
        Mean = np.nanmedian(Wband[ranges[item]:ranges[item+1] , :])
        STD = np.nanstd(Wband[ranges[item]:ranges[item+1] , :])
        Id1 = np.where(Wband[ranges[item]:ranges[item+1], :] < Mean- 3*STD)
        a[Id1] = np.NaN
        Id2 = np.where(Wband[ranges[item]:ranges[item+1], :] > Mean+ 3*STD)
        a[Id2] = np.NaN
        msk1[ranges[item]:ranges[item+1], :] = a
            
    mask = msk.reshape(r,c) * msk1.reshape(r,c) * mask
elif mask_input==2:
    mskBand = 50 
    Wband = ((data[:,:,mskBand] + data[:,:,mskBand+1])/2).reshape(data.shape[0:2])
    mask = np.ones(np.shape(Wband))
    ranges = np.linspace(0, Wband.shape[0],Wband.shape[0], dtype = 'int')    
    for item in range(Wband.shape[0]-1):
        a  = np.ones((ranges[item+1]-ranges[item], Wband.shape[1]))
        Mean = np.nanmedian(Wband[ranges[item]:ranges[item+1] , :])
        STD = np.nanstd(Wband[ranges[item]:ranges[item+1] , :])
        Id1 = np.where(Wband[ranges[item]:ranges[item+1], :] < Mean- 2.5*STD)
        a[Id1] = np.NaN
        Id2 = np.where(Wband[ranges[item]:ranges[item+1], :] > Mean+ 2*STD)
        a[Id2] = np.NaN
        mask[ranges[item]:ranges[item+1], :] = a
    
    mskBand = 850         
    Wband = ((data[:,:,mskBand] + data[:,:,mskBand+1])/2).reshape(data.shape[0:2])
    mask1 = np.ones(np.shape(Wband))
    ranges = np.linspace(0, Wband.shape[0],Wband.shape[0], dtype = 'int')    
    for item in range(Wband.shape[0]-1):
        a  = np.ones((ranges[item+1]-ranges[item], Wband.shape[1]))
        Mean = np.nanmedian(Wband[ranges[item]:ranges[item+1] , :])
        STD = np.nanstd(Wband[ranges[item]:ranges[item+1] , :])
        Id1 = np.where(Wband[ranges[item]:ranges[item+1], :] < Mean- 1.5*STD)
        a[Id1] = np.NaN
        Id2 = np.where(Wband[ranges[item]:ranges[item+1], :] > Mean+ 1.5*STD)
        a[Id2] = np.NaN
        mask1[ranges[item]:ranges[item+1], :] = a
    mask = mask.reshape(r,c) * mask1.reshape(r,c)
else:
     mask = np.ones((r,c))    

if r*c %2:
    data = data[:-1 , :,:] 
    mask = mask[:-1 , :] 
    r = r-1
data = data.reshape(r*c , b)
data[:,690:] = data[:,690:] - (data[:,690] -data[:,689]   ).reshape(-1 ,1)


data = savgol_filter(data, 7, 2)
a = np.nanmean(data.reshape(r,c,b),1)[:,:689]
b1 = scipy.signal.medfilt(a , 15)
diff = np.nanmean(np.abs(a-b1), 1)
stda = np.nanstd(diff)
meana = np.nanmean(diff)
IDs1 = np.where(diff > meana + stda)[0]
IDs2 = np.where(diff < meana - stda)[0]
IDs = np.concatenate((IDs1 , IDs2))
mask[IDs, :] = np.nan

data = (data - np.mean(data, 1, keepdims= True)) / np.std(data, 1, keepdims= True)
if data.shape[0] %2:
    data = data[:-1 , :] 
n_components = np.min(data.shape)
pca = PCA(n_components=n_components, svd_solver='full')
pca.fit(data)
samplePC = pca.transform(data)
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
data = pca.inverse_transform(samplePC)
GrainFianl = []
n_members = 1


AE =1
if AE == 1:
    print('AE')
    filename = '\CAE_encoder_model.h5'
    encoder_disck = load_model(filename)
    data = encoder_disck.predict([data.reshape(data.shape[0], 960, 1)]).squeeze()

else:
    bands = pd.read_excel(r'C:\hamid\Projects\grain size in sediment core_AnalySize\CNN\CNN results2.xlsx', sheet_name= 'Bands RF').values.flatten()
    data = data[:,bands].squeeze()

  
cnn =1
if cnn:
    print("Load MPS_CNN model from disk")
    filename = '\MPS_CNN_model.h5'
    best_model = load_model(filename)            
else:
    saved_model = pickle.load(open(r'\CAE_RF_model.sav' , 'rb'))
    best_model = saved_model['model']
        
IDs_inf = np.where(data.sum(1) == np.inf)
IDs_nan = np.where(np.isnan(data.sum(1)))
ids = np.intersect1d( np.where(data.sum(1) != np.inf) ,  np.where(~np.isnan(data.sum(1))))
data[IDs_inf] = np.mean(data[ids.reshape(1,-1) , :],1)
data[IDs_nan] = np.mean(data[ids.reshape(1,-1) , :],1)
Graindata_RF = best_model.predict(data)
Graindata_RF[IDs_inf] = np.nan
Graindata_RF[IDs_nan] = np.nan
Graindata_RF = Graindata_RF.reshape(r,c)
mask1 = np.copy(mask)
mask1[np.where(np.isnan(mask1))] =0  
Graindata_RF = Graindata_RF *mask
GrainFianl.append(Graindata_RF)
print('model result')
GrainFianl= np.nanmedian(np.array(GrainFianl), axis=0, overwrite_input=True)

Ref_data = 1
if Ref_data:
    Grain_ref = pd.read_excel(r'Grain size data Reference.xlsx', sheet_name= f'{name}').values 
    length = Grain_ref[-1,0]+.5
    Grain_ref = Grain_ref[0:-1,:]
    ref = Grain_ref[:,0]

else:
    length = 28.9 #the length of the core
    ref = np.linspace(0.5 , np.floor(length)-0.5 , int(np.floor(length)))
interval = 1
ranges = np.zeros((ref.shape[0]), dtype = 'int')    
y = np.linspace(0, length , r)
for item in range(ref.shape[0]):
    ranges[item] = np.where(np.abs(y-ref[item]) == np.min(np.abs(y-ref[item])))[0][0]


sampled_RF = np.zeros(np.size(ref))
sampled_label = np.zeros(np.size(ref))
# label = labels[:,0].reshape(r,c)
for counters in range(np.size(ref)):
    sampled_RF[counters] = np.nanmean(GrainFianl[ranges[counters]-res :ranges[counters]+res,:])
IDs = np.where(~np.isnan(sampled_RF))
sampled_RF = sampled_RF[IDs]
sampled_label = sampled_label[IDs]
ref = ref[IDs]
sampleTest_RF_high = np.nanmean(GrainFianl , axis = 1)
IDs1 = np.where(~np.isnan(sampleTest_RF_high))
sampleTest_RF_high = sampleTest_RF_high[IDs1]
y = y[IDs1]


if Ref_data:
    sampling = Grain_ref[:,1]
    sampling = sampling[IDs]
    print(names)
    r2 = r2_score( sampling , sampled_RF)
    print(r2)
    print(np.corrcoef( sampling , sampled_RF)[0][1])
    rmse = np.sqrt(mean_squared_error(sampling , sampled_RF))
    # rmse = np.sqrt(np.sum(np.power(sampling - sampled_RF, 2))) / len(sampling)
    print(rmse)
    mre = mean_relative_error(sampling , sampled_RF)
    print(mre)
    q1 = np.percentile(sampled_RF , 25)
    q3 = np.percentile(sampled_RF , 75)
    print((q3-q1) /rmse)
    fig, ax = plt.subplots(1,2, figsize=(3,8), sharey=True, sharex=False);fig.suptitle(f'{name} lake  (r = {"{:.2f}".format(np.corrcoef( sampling , sampled_RF)[0][1])})')
    ax[1].invert_yaxis()
    ax[1].plot( sampleTest_RF_high, y,color = 'orange', alpha = 0.8, linewidth=1);
    ax[1].plot(sampled_RF, ref,color = 'blue',alpha = 1, linewidth=1.2); ax[1].set_title('HSI-inferred particle size')
    ax[1].tick_params(axis='both', which='major', labelsize=15);
    ax[1].set_xlabel('Particle size (\u03BCm )');
    ax[0].plot(sampling, ref,color = 'black',linewidth=1); ax[0].set_title('Reference Grain size')  
    ax[0].plot( savgol_filter(sampleTest_RF_high, 15, 3), y,color = 'orange', alpha = 0.8, linewidth=1);
    ax[0].tick_params(axis='both', which='major', labelsize=15)

else:
    fig, ax = plt.subplots(1,1, figsize=(3,8), sharey=True, sharex=True);fig.suptitle(f'{name} lake')
    ax.invert_yaxis()
    ax.plot( sampleTest_RF_high, y,color = 'orange', alpha = 0.5, linewidth=0.5);
    ax.plot( savgol_filter(sampleTest_RF_high, 15, 3), y,color = 'blue', alpha = 1, linewidth=1);
    ax.tick_params(axis='both', which='major', labelsize=9);
    ax.set_xlabel('Particle size (\u03BCm )');
    ax.set_ylabel('Dpeth (cm)')

end = time.time()
print("Elappsed Time" , end - start , "seconds")

import os
import random
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam



def get_filename(filepath):
    return filepath.split('\\')[-1]


def importDataInfo(path):
    columns=['center','left','right','Steering','Throttle','Brake','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    data['center']=data['center'].apply(get_filename)

    return data

def BalanceData(data,display=True):
    nBins=31
    samplePerBin=750
    hist,bins=np.histogram(data['Steering'],nBins)
    if display:
        center=(bins[:-1]+bins[1:])*0.5

        plt.bar(center,hist,width=0.08)
        plt.plot((-1,1),(samplePerBin,samplePerBin))
        plt.show()

    removeIndexList=[]
    for j in range(nBins):
        binDataList=[]
        for i in range(len(data['Steering'])):
            if data['Steering'][i]>=bins[j] and data['Steering'][i]<=bins[j+1]:
                binDataList.append(i)

        binDataList=shuffle(binDataList)
        binDataList=binDataList[samplePerBin:]
        removeIndexList.extend(binDataList)

    data.drop(data.index[removeIndexList],inplace=True)
    print('Total Images',len(data))
    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center,hist,width=0.08)
        plt.plot((-1,1),(samplePerBin,samplePerBin))
        plt.show()

    return data

def LoadData(path,data):
    imagespath=[]
    steering=[]

    for i in range(len(data)):
        indexedData=data.iloc[i]
        imagespath.append(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))

    imagespath=np.asarray(imagespath)
    steering=np.asarray(steering)

    return imagespath,steering

def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

def preProcess(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255

    return img

def BatchGen(imagesPath,SteeringList,batch_size,trainFlag):
    while True:
        imageBatch=[]
        SteerBatch=[]

        for i in range(batch_size):
            index=random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img,steer=augmentImage(imagesPath[index],SteeringList[index])
            else:
                img=mpimg.imread(imagesPath[index])
                steer=SteeringList[index]
            img=preProcess(img)

            imageBatch.append(img)
            SteerBatch.append(steer)
        yield (np.asarray(imageBatch),np.asarray(SteerBatch))


def createModel():
    model=Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2),activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2),activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss='mse')

    return model




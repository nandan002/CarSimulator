print('And It starts Now!!')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from Utilities import *
from sklearn.model_selection import train_test_split
path='Data'
data=importDataInfo(path)

data=BalanceData(data,display=False)

imagepath,steering=LoadData(path,data)

xTrain,xVal,yTrain,yVal=train_test_split(imagepath,steering,test_size=0.2,random_state=5)

print("Total",len(xTrain),len(xVal))

model=createModel()
model.summary()

# TRAINING MODEL
history=model.fit(BatchGen(xTrain,yTrain,128,1),steps_per_epoch=300,epochs=10,validation_data=BatchGen(xVal,yVal,128,0),validation_steps=200)

model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim([0,1])
plt.legend(['Loss','Val Loss'])
plt.show()

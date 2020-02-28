#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Any results you write to the current directory are saved as output.
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LeakyReLU
import numpy as np
import os
import h5py
#from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA



# In[2]:


# In[4]:


def find(SNR):
    #H_R.shape = (512,56,924,5)
    
    #print(temp.shape)
    idx  = np.argmax(SNR, axis=2)
    return idx
    
def preprocess(H_Re, idx):
    #temp = np.zeros(H_Re.shape[:-1])
    for i in range(H_Re.shape[0]):
        for j in range(H_Re.shape[1]):
            
            H_Re[i,j,:,0] = H_Re[i,j,:,idx[i,j]]
    return H_Re[:,:,:,0]

def preprocess2(H_Re, idx):
    temp = np.zeros(H_Re.shape[:-1])
    for i in range(H_Re.shape[0]):
        for j in range(H_Re.shape[1]):
            temp[i,j] = H_Re[i,j,idx[i,j]]
    return temp
      
        


# In[6]:


def get_data(data_file):
    f = h5py.File(data_file, 'r')
    H_Re = f['H_Re'][:] #shape (sample size, 56, 924, 5)
    H_Im = f['H_Im'][:] #shape (sample size, 56, 924, 5)
    SNR = f['SNR'][:] #shape (sample size, 56, 5)
    Pos = f['Pos'][:] #shape(sample size, 3)
    f.close()
    return H_Re, H_Im, SNR, Pos


# In[3]:


CTW_labelled = "/kaggle/input/ctw2020/"
data_file = CTW_labelled+"file_"+str(1)+".hdf5"
H_Re, H_Im, SNR, Pos = get_data(data_file)

#print(H_Re[1,1,1,:])

idx = find(SNR)

H_Re = preprocess(H_Re, idx)
H_Im = preprocess(H_Im, idx)
SNR = preprocess2(SNR, idx)
print(SNR.shape)


# In[4]:


# In[7]:

for i in range(2,3):
    temp = CTW_labelled + "file_"+str(i)+".hdf5"
    tH_Re, tH_Im, tSNR, tPos = get_data(temp)
    idx = find(tSNR)
    tH_Re = preprocess(tH_Re, idx)
    tH_Im = preprocess(tH_Im, idx)
    tSNR = preprocess2(tSNR, idx)
    H_Re, H_Im, Pos, SNR  = np.concatenate((H_Re, tH_Re)), np.concatenate((H_Im, tH_Im)), np.concatenate((Pos, tPos)), np.concatenate((SNR, tSNR))


# In[5]:


# In[8]:


#print(H_Re[:,1,1])

samples = H_Re.shape[0]
H_Re = H_Re.reshape((samples,-1))
print(H_Re.shape)
H_Im = H_Im.reshape((samples,-1))
#SNR = SNR.reshape((samples,-1))
#Pos = Pos.reshape((samples,-1))

data = np.concatenate((H_Re, H_Im, SNR), axis=1)
print(data.shape)


# In[ ]:


# In[ ]:


'''

model = Sequential()
model.add(Dense(compression_1, activation = 'relu', input_shape=(517718,), name = 'compress'))
model.add(Dense(517718, activation='linear'))
model.compile(loss= 'mean_squared_error', optimizer = 'adam')
#model.loadweights('../input/weights_1/saved_model.pb')
#print('model summary')
'''


# In[25]:


# In[9]:

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adadelta
adal = Adadelta(learning_rate=0.5, rho=0.95)
def create_model():
    model = Sequential([ Dense(200 ,input_shape=(103544,)), LeakyReLU(alpha=0.2)
                           ,Dense(103544, activation='linear')])

    model.compile(loss= 'mean_squared_error', optimizer = adal) 
    return model
print(data.shape)


# In[26]:


model = create_model()
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopper = EarlyStopping(patience = 30, verbose=1)
checkpointer = ModelCheckpoint('Best', verbose=1, save_best_only=True)
results = model.fit(data[:1000], data[:1000], validation_split=0.25, batch_size = 1, epochs = 100, callbacks=[earlystopper, checkpointer])
model.save('mymodel.h5')


# In[17]:


# In[12]:
print(data[1020])
print(np.sum(((model.predict(data[1020:1021]))-data[1020])**2)/103544)


# In[ ]:


# In[ ]:


results = model.fit(data[int(data.shape[0]/5):2*int(data.shape[0]/5)], data[int(data.shape[0]/5):2*int(data.shape[0]/5)], validation_split=0.1, batch_size = 10, epochs = 100, callbacks=[earlystopper, checkpointer])


# In[ ]:


# In[ ]:
results = model.fit(data[2*int(data.shape[0]/5):3*int(data.shape[0]/5)], data[2*int(data.shape[0]/5):3*int(data.shape[0]/5)], validation_split=0.1, batch_size = 10, epochs = 100, callbacks=[earlystopper, checkpointer])


# In[27]:


# In[ ]:

c_model = create_model()
#print(os.listdir("../output"))

c_model = load_model('/kaggle/working/Best')


# In[29]:


# In[ ]:
k=1021
X_decompressed = c_model.predict(data[1000:])
print(np.sum((X_decompressed - data[1000:])**2)/(103544*24))


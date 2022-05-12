#!/usr/bin/env python
# coding: utf-8

# # Germany Speech Emotion Recognition

# #### Importing the neccessary libraries

# In[1]:


import numpy as np
import pandas as pd
from librosa import display
import os
import librosa
import IPython.display as ipd
import glob
import sys
import matplotlib.pyplot as plt


# #### Getting the path of current directory

# In[2]:


path = os.getcwd()
print(path)


# In[3]:


Germany_data = "/home/thanuja/Downloads/multi-lang emotion data/wav/"


# In[4]:


Germany_data_list = os.listdir(Germany_data)
Germany_data_list


# In[5]:


emotion=[]
path=[]

for files in Germany_data_list:
    if files[5:6]=='E':
        emotion.append('disgust')
    elif files[5:6]=='W':
        emotion.append('anger')
    elif files[5:6]=='L':
        emotion.append('boredom')
    elif files[5:6]=='A':
        emotion.append('fear')
    elif files[5:6]=='F':
        emotion.append('happy')
    elif files[5:6]=='T':
        emotion.append('sad')
    elif files[5:6]=='N':
        emotion.append('Neutral')
    else:
        emotion.append('Error')
    path.append(Germany_data+files)


# In[6]:


data_emo = pd.DataFrame(emotion,columns=['Emotion'])
data_emo


# In[7]:


data_path = pd.DataFrame(path,columns=['Path'])
pd.set_option('display.max_colwidth',None)
data_path


# In[8]:


data = pd.concat([data_path,data_emo],axis=1)
data


# In[9]:


#Function to create waveplot
def createWaveplot(data, sr, e):
    plt.figure(figsize=(10,3))
    plt.title('Waveplot for audio with {} emotion'.format(e),size=15)
    librosa.display.waveplot(data, sr=sr)
    plt.show()


# In[10]:


#Function to create spectrogram plot
def createSpectrogram(data, sr, e):
    X= librosa.stft(data)
    Xdb=librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12,3))
    plt.title('Spectrogram for audio with {} emotion'.format(e),size=15)
    librosa.display.specshow(Xdb, sr=sr,x_axis='time',y_axis='hz')
    plt.colorbar()


# In[11]:


#Visualising the data with both the plots and playing the audio
emotion = 'anger'
path =np.array(data.Path[data.Emotion==emotion])[4]
audio_data_ravdess,sampling_rate1 = librosa.load(path)

createWaveplot(audio_data_ravdess,sampling_rate1,emotion)
createSpectrogram(audio_data_ravdess,sampling_rate1,emotion)
ipd.Audio(path)


# In[12]:


#Data = data.reindex(np.random.permutation(data.index))
#Data


# In[13]:


#Data.to_csv("GData1.csv",index=False)


# In[14]:


speech_data=pd.read_csv("GData1.csv")
speech_data


# In[15]:


Feature_data = pd.DataFrame(columns=['Features'])

counter = 0
for index, path in enumerate(speech_data.Path):
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
    Feature_data.loc[counter]=[mfccs]
    counter=counter+1


# In[16]:


Feature_data.head()


# In[17]:


Feature_data=pd.concat([pd.DataFrame(Feature_data['Features'].values.tolist()),speech_data.Emotion],axis=1)
Feature_data


# In[18]:


X_data = Feature_data.drop(['Emotion'], axis=1)
X_data


# In[19]:


Y_data = Feature_data.Emotion
Y_data.head()


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=40)


# In[21]:


print((x_train.shape,x_test.shape,y_train.shape,y_test.shape))


# In[22]:


from sklearn.svm import SVC
svc_model=SVC(kernel='rbf').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[23]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# In[24]:


from sklearn.svm import SVC
svc_model=SVC(gamma=0.01,kernel='linear').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[25]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# In[26]:


from sklearn.svm import SVC
svc_model=SVC(kernel='poly').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[27]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# In[28]:


final_model_germany =SVC(kernel='poly').fit(X_data,Y_data)


# In[29]:


import pickle

pickle.dump(final_model_germany,open('germany.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





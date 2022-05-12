#!/usr/bin/env python
# coding: utf-8

# # Italian Speech Emotion Recognition

# #### Importing the necessary libraries

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


# #### Setting the path where the speech data is present

# In[3]:


Italian_data = "/home/thanuja/Downloads/multi-lang emotion data/EMOVO/"


# #### Listing the sub-folders present

# In[4]:


Italian_data_list = os.listdir(Italian_data)
Italian_data_list


# #### Extracting and Saving the emotion and path of the data

# In[5]:


emotion=[]
path=[]

for sub_dir in Italian_data_list:
    filename = os.listdir(Italian_data+sub_dir)
    for files in filename:
        part = files.split('.')[0].split('-')
        emotion.append(str(part[0]))
        path.append(Italian_data+sub_dir+'/'+files)
        


# #### Converting the saved emotion list to dataframe

# In[6]:


data_emo = pd.DataFrame(emotion,columns=['Emotion'])
data_emo.head()


# #### Replacing the Italian emotion denotion to english

# In[7]:


data_emo1 = data_emo.replace({'neu':'neutral','sor':'surprise','pau':'fear',
                              'tri':'sad','goi':'joy','rab':'angry','dis':'disgust'})


# #### Converting the saved path list to dataframe and viewing it in full form

# In[8]:


data_path = pd.DataFrame(path,columns=['Path'])
pd.set_option('display.max_colwidth',None)
data_path


# #### Concatenating both the path and the emotion

# In[9]:


data = pd.concat([data_path,data_emo1],axis=1)
data


# #### Creating two functions to plot the audio data

# In[10]:


#Function to create waveplot
def createWaveplot(data, sr, e):
    plt.figure(figsize=(10,3))
    plt.title('Waveplot for audio with {} emotion'.format(e),size=15)
    librosa.display.waveplot(data, sr=sr)
    plt.show()


# In[11]:


#Function to create spectrogram plot
def createSpectrogram(data, sr, e):
    X= librosa.stft(data)
    Xdb=librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12,3))
    plt.title('Spectrogram for audio with {} emotion'.format(e),size=15)
    librosa.display.specshow(Xdb, sr=sr,x_axis='time',y_axis='hz')
    plt.colorbar()


# In[12]:


#Visualising the data with both the plots and playing the audio
emotion = 'angry'
path =np.array(data.Path[data.Emotion==emotion])[2]
audio_data_ravdess,sampling_rate1 = librosa.load(path)

createWaveplot(audio_data_ravdess,sampling_rate1,emotion)
createSpectrogram(audio_data_ravdess,sampling_rate1,emotion)
ipd.Audio(path)


# ####  Shuffling the data and saving in the csv format

# In[13]:


#Data = data.reindex(np.random.permutation(data.index))
#Data


# In[14]:


#Data.to_csv("IData1.csv",index=False)


# In[15]:


speech_data=pd.read_csv("IData1.csv")
speech_data


# #### Extracting the features

# In[16]:


Feature_data = pd.DataFrame(columns=['Features'])

counter = 0
for index, path in enumerate(speech_data.Path):
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
    Feature_data.loc[counter]=[mfccs]
    counter=counter+1


# #### Viewing the extracted features

# In[17]:


Feature_data.head()


# #### Concatenating the features and emotion and making the features into column form

# In[18]:


Feature_data=pd.concat([pd.DataFrame(Feature_data['Features'].values.tolist()),speech_data.Emotion],axis=1)
Feature_data


# #### Independent data

# In[19]:


X_data = Feature_data.drop(['Emotion'], axis=1)
X_data


# #### Dependent or target data

# In[20]:


Y_data = Feature_data.Emotion
Y_data.head()


# #### Train and Test split

# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=40)


# In[22]:


print((x_train.shape,x_test.shape,y_train.shape,y_test.shape))


# ### Model Building

# #### Support Vector Classifier - Radial Basis Function Kernel

# In[23]:


from sklearn.svm import SVC
svc_model=SVC(gamma=0.00001,kernel='rbf').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[24]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### Support Vector Classifier- Linear Function Kernel

# In[25]:


from sklearn.svm import SVC
svc_model=SVC(kernel='linear').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[26]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### Support Vector Machine - Polynomial function Kernel

# In[27]:


from sklearn.svm import SVC
svc_model=SVC(C=1000,kernel='poly').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[28]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# In[29]:


final_model_italian=SVC(C=1000,kernel='poly').fit(X_data,Y_data)


# In[30]:


import pickle

pickle.dump(final_model_italian,open('italian.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





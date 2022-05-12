#!/usr/bin/env python
# coding: utf-8

# # Greek Speech Emotion Recognition

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


# #### Getting the path of  current directory

# In[2]:


path = os.getcwd()
print(path)


# #### Setting the path of the speech data present

# In[3]:


Greek_data = "/home/thanuja/Downloads/multi-lang emotion data/Acted Emotional Speech Dynamic Database/"
Greek_data


# #### Listing the sub-folders present

# In[4]:


Greek_data_list = os.listdir(Greek_data)
Greek_data_list


# In[5]:


#saving the path and emotion of the data within the respective variables

emotion=[]
path=[]

for sub_dir in Greek_data_list:
    filename = os.listdir(Greek_data+sub_dir)
    for files in filename:
        if files[0:1]=='a':
            emotion.append('angry')
        elif files[0:1]=='d':
            emotion.append('disgust')
        elif files[0:1]=='f':
            emotion.append('fear')
        elif files[0:1]=='h':
            emotion.append('happy')
        elif files[0:1]=='s':
            emotion.append('sad')
        else:
            emotion.append('Error')
        path.append(Greek_data+sub_dir+'/'+files)


# #### Converting the saved emotion list to dataframe

# In[6]:


data_emo = pd.DataFrame(emotion, columns=['Emotion'])
data_emo.shape


# #### Converting the saved path list to dataframe and viewing it in full form

# In[7]:


data_path = pd.DataFrame(path,columns=['Path'])
pd.set_option('display.max_colwidth', None)
data_path


# #### Concatenating both the path and the emotion

# In[8]:


data = pd.concat([data_path,data_emo], axis=1)
data


# #### Creating two functions to plot the audio data

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
emotion = 'disgust'
path =np.array(data.Path[data.Emotion==emotion])[6]
audio_data_ravdess,sampling_rate1 = librosa.load(path)

createWaveplot(audio_data_ravdess,sampling_rate1,emotion)
createSpectrogram(audio_data_ravdess,sampling_rate1,emotion)
ipd.Audio(path)


# ####  Shuffling the data and saving in the csv format

# In[43]:


#Data = data.reindex(np.random.permutation(data.index))
#Data


# In[44]:


#Data.to_csv("Data1.csv",index=False)


# In[12]:


speech_data=pd.read_csv("Data1.csv")
speech_data


# #### Extracting the features

# In[13]:


Feature_data = pd.DataFrame(columns=['Features'])

counter = 0
for index, path in enumerate(speech_data.Path):
    X, sample_rate = librosa.load(path,res_type='kaiser_fast')
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,n_mfcc=40).T, axis=0)
    Feature_data.loc[counter]=[mfccs]
    counter=counter+1


# #### Viewing the extracted features

# In[14]:


Feature_data.head()


# #### Concatenating the features and emotion and making the features into column form

# In[15]:


Feature_data=pd.concat([pd.DataFrame(Feature_data['Features'].values.tolist()),speech_data.Emotion],axis=1)
Feature_data


# #### Independent data

# In[16]:


X_data = Feature_data.drop(['Emotion'], axis=1)
X_data


# #### Dependent or target data

# In[17]:


Y_data = Feature_data.Emotion
Y_data.head()


# #### Train and Test split

# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=40)


# In[19]:


print((x_train.shape,x_test.shape,y_train.shape,y_test.shape))


# ### Model Building

# #### Support Vector Classifier - Radial Basis Function Kernel

# In[20]:


from sklearn.svm import SVC
svc_model=SVC(kernel='rbf').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[21]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### Support Vector Classifier- Linear Function Kernel

# In[22]:


from sklearn.svm import SVC
svc_model=SVC(C=100,kernel='linear').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[23]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# #### Support Vector Machine - Polynomial function Kernel

# In[24]:


from sklearn.svm import SVC
svc_model=SVC(C=100,kernel='poly').fit(x_train,y_train)

print('accuracy: {}'.format(svc_model.score(x_test,y_test)))


# In[25]:


train_acc = float(svc_model.score(x_train, y_train)*100)
print("----train accuracy score %s ----" % train_acc)

test_acc = float(svc_model.score(x_test, y_test)*100)
print("----test accuracy score %s ----" % test_acc)


# In[26]:


final_model_greek =SVC(C=100,kernel='poly').fit(X_data,Y_data)


# In[28]:


import pickle

pickle.dump(final_model_greek,open('greek.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





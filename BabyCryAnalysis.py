#!/usr/bin/env python
# coding: utf-8

# # Baby Cry Classification

# In[1]:


#Store all audio files in dictionary where key: filename, value: label
import os
raw_audio = dict()
directory = 'audio\hungry'
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        raw_audio[os.path.join(directory, filename)] = 'hungry'
    else:
        continue

directory = 'audio\pain'
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        raw_audio[os.path.join(directory, filename)] = 'pain'
    else:
        continue
        
directory = 'audio\discomfort'
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        raw_audio[os.path.join(directory, filename)] = 'discomfort'
    else:
        continue


#print raw_audio


# In[ ]:


import wave 
import math

def chop_song(filename, folder):
    handle = wave.open(filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))
    #print filename
    last_number_frames = 0
    #Slicing Audio file
    for i in range(num_secs):
        
        shortfilename = filename.split("/")[1].split(".")[0]
        snippetfilename = 'audio/' + folder + '/' + shortfilename + 'snippet' + str(i+1) + '.wav'
        #print snippetfilename
        snippet = wave.open(snippetfilename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        #snippet.setsampwidth(2)
        #snippet.setframerate(11025)
        snippet.setnframes(handle.getnframes())
        snippet.writeframes(handle.readframes(window_size))
        handle.setpos(handle.tell() - 1 * frame_rate)
        #print snippetfilename, ":", snippet.getnchannels(), snippet.getframerate(), snippet.getnframes(), snippet.getsampwidth()
        
        #The last audio slice might be less than a second, if this is the case, we don't want to include it because it will not fit into our matrix 
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            #print "this file doesnt have the same frame size!, remaming file"
            os.rename(snippetfilename, snippetfilename+".bak")
        snippet.close()

    #handle.close()

for audio_file in raw_audio:
    chop_song(audio_file, raw_audio[audio_file])


# In[ ]:


import pandas as pd
import librosa 
import numpy as np
'''Chop and Transform each track'''
X = pd.DataFrame(columns = np.arange(45), dtype = 'float32').astype(np.float32)
j = 0
k = 0
for i, filename in enumerate(os.listdir('audio/pain/')):
    last_number_frames = -1
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("audio/pain/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[44] = 'pain'
        X.loc[i] = x.loc[0]
        j = i 
        

for i, filename in enumerate(os.listdir('audio/hungry/')):
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("audio/hungry/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[44] = 'hungry'
        X.loc[i+j] = x.loc[0] 
        k = i 
        
for i, filename in enumerate(os.listdir('audio/discomfort/')):
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("audio/discomfort/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[44] = 'discomfort'
        X.loc[i+j+k] = x.loc[0]
        
#Do something with missing values. you might want to do something more sophisticated with missing values later
X = X.fillna(0)


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split

y = X[44]
del X[44]
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


#Code to hide deprication warnings

from IPython.display import HTML
HTML('''<script>
code_show_err=false; 
function code_toggle_err() {
 if (code_show_err){
 $('div.output_stderr').hide();
 } else {
 $('div.output_stderr').show();
 }
 code_show_err = !code_show_err
} 
$( document ).ready(code_toggle_err);
</script>
To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.''')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt


def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
        model = classifier(**kwargs)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        return model.score(X_test, y_test),                precision_score(y_test, y_predict, average='micro'),                recall_score(y_test, y_predict, average='micro')

print ("    Accuracy of classifiers:")
print ("    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5))
print ("    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test))
print ("    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test))
print ("    SVM:", get_scores(SVC, X_train, X_test, y_train, y_test))
#print "    Naive Bayes:", get_scores(MultinomialNB, X_train, X_test, y_train, y_test)

#Plot non-normalized confusion matrix
classifier= SVC(kernel='linear',C=0.1).fit(X_train,y_train)

titles_options=[("Confusion matrix without normalization",None),("Normalised confusion matrix",'true')]
for title,normalize in titles_options:
    disp=plot_confusion_matrix(classifier,X_train,y_train,normalize=normalize)
    disp=plot_confusion_matrix(classifier,X_test,y_test,normalize=normalize)
    
    disp.ax_.set_title(title)
    
    
    print(title)
    print(disp.confusion_matrix)
    
plt.show()


classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred,labels=['discomfort','hunger','pain']))


# In[ ]:


import pickle as cPickle

def pickle_model(model, modelname):
    with open('models/' + str(modelname) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)

model = RandomForestClassifier()
model.fit(X,y)
pickle_model(model, "myDecisionTree")


# In[ ]:


import pickle as cPickle

def getModel(pickle_path):
    with open(pickle_path, 'rb') as f:
        return cPickle.load(f)


# In[ ]:


model = getModel("models/myDecisionTree.pkl")


# In[ ]:


import wave 
import math

def chop_new_audio(filename, folder):
    handle = wave.open(filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 1 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))
    #print filename
    last_number_frames = 0
    #Slicing Audio file
    for i in range(num_secs):
        
        shortfilename = filename.split(".")[0]
        snippetfilename = folder + '/' + shortfilename + 'snippet' + str(i+1) + '.wav'
        #print snippetfilename
        snippet = wave.open(snippetfilename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        #snippet.setsampwidth(2)
        #snippet.setframerate(11025)
        snippet.setnframes(handle.getnframes())
        snippet.writeframes(handle.readframes(window_size))
        handle.setpos(handle.tell() - 1 * frame_rate)
        #print snippetfilename, ":", snippet.getnchannels(), snippet.getframerate(), snippet.getnframes(), snippet.getsampwidth()
        
        #The last audio slice might be less than a second, if this is the case, we don't want to include it because it will not fit into our matrix 
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            #print "this file doesnt have the same frame size!, remaming file"
            os.rename(snippetfilename, snippetfilename+".bak")
        snippet.close()

    #handle.close()


# In[ ]:


chop_new_audio("predict.wav", "predict")


# In[ ]:


predictions = []
for i, filename in enumerate(os.listdir('predict/')):
    last_number_frames = -1
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("predict/"+filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        prediction = model.predict(fingerprint)
        #print prediction
        predictions.append(prediction[0])


# In[ ]:


print(predictions[0])


# In[ ]:





# In[ ]:





# In[ ]:





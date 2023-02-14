
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from mlxtend.classifier import MultiLayerPerceptron as MLP 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
            
#     print(result)
    return result

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust','neutral','surprised','sad','angry']
#Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("H:\\PROJECTS\\Ml package\\speech_reg\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train,x_test,y_train,y_test=load_data(test_size=0.25)
# Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

print(f'Features extracted: {x_train.shape[1]}')
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=500)
# - Train the model
model.fit(x_train,y_train)

# Predict for the test set
y_pred=model.predict(x_test)
def speech_res():    
    return y_pred
import seaborn as sns
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
#Get the confusion matrix

# Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("ReLu-Accuracy: {:.2f}%".format(accuracy*100))
ReLu-Accuracy: 46.39%
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=500,activation='logistic')
#Train the model
model.fit(x_train,y_train)

#Predict for the test set
y_pred=model.predict(x_test)
def speech_res():    
    return y_pred
import seaborn as sns
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
#Get the confusion matrix

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Logistic-Accuracy: {:.2f}%".format(accuracy*100))
Logistic-Accuracy: 60.56%
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=500,activation='tanh')
#Train the model
model.fit(x_train,y_train)

#Predict for the test set
y_pred=model.predict(x_test)
def speech_res():    
    return y_pred
import seaborn as sns
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
#Get the confusion matrix

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Tanh-Accuracy: {:.2f}%".format(accuracy*100))
Tanh-Accuracy: 58.89%
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#Fit the model
logreg = LogisticRegression(C=1e5)
logreg.fit(x_train,y_train)
#Generate predictions with the model using our X values
y_pred = logreg.predict(x_test)
#Get the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Logistic-Accuracy: {:.2f}%".format(accuracy*100))

import seaborn as sns
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
#Get the confusion matrix


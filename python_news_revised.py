#     IMPORTING BASIC LIBRARIES

# importing basic library
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score

# importing libraries of natural language processing
import re
import nltk
#nltk.download('stopwords')      # download stopword from nltk like a,an,the,etc
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#     DATASET AND DECLARING X AND Y 

dataset=pd.read_csv('news.csv')
X=dataset.iloc[:,0:3].values
y=dataset.iloc[:,-1].values

#     CLEANING OF DATA

corpus=[]
zz=[]
for i in range(0,6335):
    a1=re.sub('[^a-zA-Z]',' ',dataset['title'][i])
    a2=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    a1=a1.lower()
    a2=a2.lower()
    a1=a1.split()
    a2=a2.split()
    ps1=PorterStemmer()
    ps2=PorterStemmer()
    a1 = [ps1.stem(word) for word in a1 if not word in set(stopwords.words('english'))]
    a1 = ' '.join(a1)
    a2 = [ps2.stem(word) for word in a2 if not word in set(stopwords.words('english'))]
    a2 = ' '.join(a2)
    a=a1+a2
    corpus.append(a)
zz.append(corpus)  

 
#   ENCODING FAKE WITH 0 AND REAL WITH 1 IN Y VECTOR

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)


#   CREATING A BAG OF WORD >> A SPARSE MATRIX OF SIZE row * 1500 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()


#   Train-Test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''

Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')        # 79.95%

'''
'''
kNearestNeighbour classifier

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=1)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')     # n_neighbour(10=80.9%)

'''

'''
Support Vector Machine
from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')    # 87.61 %

'''

'''
DecisionTreeClassification

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%') # 80.9%

'''
'''

RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion='gini')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')  # 90.61%
'''

'''
passiveAggresiveClassifier

from sklearn.linear_model import PassiveAggressiveClassifier
classifier=PassiveAggressiveClassifier(max_iter=1000)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')  # 88.79%
'''

#             APPLYING ARTIFICIAL NEURAL NETWORK


# BASIC LIBRARY FOR ANN

import keras
from keras.models import Sequential      # sequential class to initialize ann
from keras.layers import Dense,Dropout        # dense class to give layer


#  initialing the ann model
def model():
    classifier=Sequential()
    
    
    # adding 1st hidden layer with 8 perceptrons(ie neurons) and activation fun is relu (rectilinear )
    classifier.add(Dense(units=6,activation='elu')) 
    classifier.add(Dropout(0.05))    
    # add 2nd hidden layer
    classifier.add(Dense(units=6,activation='elu')) 
    classifier.add(Dropout(0.05))
    classifier.add(Dense(units=6,activation='elu')) 
    classifier.add(Dropout(0.05))

    
    # final output layer with 1 output and activation fun sigmoid
    classifier.add(Dense(units=1,activation='sigmoid'))
    
    
    
    #  COMPILING RHE ANN ,LOSS=BINARY_CROSSENTROPY AS OUTPUT IS BINARY ,METRICS=ACCURACY
    classifier.compile(optimizer='adam' ,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

#   FITTING THE MODEL IN TRAIN SET AND NO OF EPOCHS =100 (ie machine with backpropagate 100 time)
classifier=model()
classifier.fit(X_train,y_train,batch_size=10,epochs=100)


classifier.summary()

#        P.S.->>>>   ACCURACY OF TRAINING SET IS 1.0000




#   PRIDICTION OF TEST RESULT

y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)     # Y_PRED IS A FLOAT OF IF <0.5 -> 0 ELSE 1

#   MAKING CONFUSION MATRIX AND PRDICTIONG ACCURACCY SCORE ON TEST SET 


cm= confusion_matrix(y_test, y_pred)

score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')     # 91.16%


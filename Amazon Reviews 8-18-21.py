# -*- coding: utf-8 -*-
"""
Created on August 18, 2021

@author: Marcel Bodevin

Georgetown Data Science Certificate Program
"""

import time
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

StartTime = time.time()

PetSuppliesDF = pd.read_json('Pet_Supplies_5.json', lines=True) # Read in json file as a dataframe
PetSuppliesDF = PetSuppliesDF.sample(frac = 0.1) #To help with intense computational requirements, limiting to 10% of data (100,000 reviews)

#Add column with Date from converted Unix time. Unfortunately results does not give time.
PetSuppliesDF["Date"] = pd.to_datetime(PetSuppliesDF["unixReviewTime"], unit='s')

#Create binary rating column: 0 (negative), 1 (positive)
conditions = [
    (PetSuppliesDF["overall"] > 2),
    (PetSuppliesDF["overall"] < 3)
    ]
values = [1, 0]
PetSuppliesDF['BinaryRating'] = np.select(conditions, values)

#Create column of review text with all lowercase, no punctuation, and no stopwords
nan_value = float("NaN") #Create na variable for blanks
PetSuppliesDF["reviewText"].replace("", nan_value, inplace=True) #Replace blanks with na variable
PetSuppliesDF.dropna(subset = ["reviewText"], inplace=True) #Drop all rows with na review text
PetSuppliesDF["ReviewNoFiller"] = PetSuppliesDF["reviewText"].str.replace('[^\w\s]','',regex=True) #Create column with review text with no punctuation
PetSuppliesDF["ReviewNoFiller"] = PetSuppliesDF["ReviewNoFiller"].str.lower() #Make all words lowercase
stopwords = stopwords.words('english') #Create stopwords variable
PetSuppliesDF["ReviewNoFiller"] = PetSuppliesDF["ReviewNoFiller"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) #Remove stop words
PetSuppliesDF["ReviewNoFiller"].replace("", nan_value, inplace=True,regex=True) #Replace blanks with na
PetSuppliesDF.dropna(subset = ["ReviewNoFiller"], inplace=True) #Drop all rows with na review text

#Insert columns with tokenized review and summary
PetSuppliesDF["ReviewToken"] = PetSuppliesDF.apply(lambda row: word_tokenize(row["ReviewNoFiller"]), axis=1)

#Lemmatize all reviews and summaries, rejoin the strings
WNL = WordNetLemmatizer()
def lemmatize_text(text):
    return [WNL.lemmatize(w) for w in text]
PetSuppliesDF['ReviewLemma'] = PetSuppliesDF.ReviewToken.apply(lemmatize_text)
PetSuppliesDF['ReviewLemma'] = PetSuppliesDF['ReviewLemma'].apply(' '.join)

#Print out distribution of resulting review ratings
print(PetSuppliesDF['overall'].value_counts())

#Insert column with VADER sentiment analysis compound score of full review text, scale numbers from 1 to 5
vader = SentimentIntensityAnalyzer()
PetSuppliesDF["VaderCompound"] = [vader.polarity_scores(x)['compound'] for x in PetSuppliesDF["reviewText"]]
scaler = MinMaxScaler(feature_range=(1,5))
PetSuppliesDF["VaderCompound"] = scaler.fit_transform(PetSuppliesDF["VaderCompound"].values.reshape(-1,1))

#Insert column with review word count
PetSuppliesDF["WordCount"] = PetSuppliesDF["ReviewToken"].apply(len)

#What does word count distribution look like? Need visualization to decide how to bin data. Also look at descriptive statistics.
WordHist = PetSuppliesDF.hist(column = 'WordCount', bins=300)
plt.xlim([0,150])
print(PetSuppliesDF["WordCount"].describe()) #25% is 6 or less, 25% is 29 words or more, will bin accordingly

#Create column categorizing review word count as short (1) or not (0)
conditions = [
    (PetSuppliesDF["WordCount"] < 7),
    (PetSuppliesDF["WordCount"] > 6)
    ]
values = [1,0]
PetSuppliesDF['Short'] = np.select(conditions, values)

#Create column categorizing review word count as long (1) or not (0)
conditions = [
    (PetSuppliesDF["WordCount"] > 28),
    (PetSuppliesDF["WordCount"] < 29)
    ]
values = [1,0]
PetSuppliesDF['Long'] = np.select(conditions, values)

#Create column categorizing reviewer as verified (1) or not (0)
conditions = [
    (PetSuppliesDF['verified'] == True),
    (PetSuppliesDF['verified'] == False)
    ]
values = [1, 0]
PetSuppliesDF['Verified'] = np.select(conditions, values)

#Create binary column if the reviewer uploaded an image (1) or did not (0)
conditions = [
    (pd.notnull(PetSuppliesDF['image'])),
    (pd.isnull(PetSuppliesDF['image']))
    ]
values = [1, 0]
PetSuppliesDF['IsImage'] = np.select(conditions, values)

#Adjust vote column to allow for analysis
print(PetSuppliesDF.dtypes,'\n')
PetSuppliesDF['vote'] = PetSuppliesDF['vote'].str.replace('[^\w\s]','',regex=True) #Remove all punctuation from strings
PetSuppliesDF['vote'].replace('', '0', inplace=True) #Replace blanks with 0
PetSuppliesDF['vote'] = PetSuppliesDF['vote'].fillna('0') # Replace na values with 0
PetSuppliesDF['vote'] = PetSuppliesDF['vote'].astype({'vote': 'int32'})
print(PetSuppliesDF.dtypes,'\n')

#Adjust data frame for analysis, dropping unneccesary columns
PetSuppliesDF = PetSuppliesDF[['ReviewLemma','VaderCompound','Short','Verified','Long','IsImage','WordCount','vote','overall','BinaryRating']]

# Write final dataframe into csv
PetSuppliesDF.to_csv(r'PetSupplies.csv', index = False)

#Print some of the dataframe to verify work
pd.set_option('display.max_columns', None) #So as not to truncate output
pd.set_option('display.max_rows', None) #So as not to truncate output
for col in PetSuppliesDF.columns: #Print column names
    print(col)
print(PetSuppliesDF.head()) # Print first five entries in dataframe

"""
First group of models are binary models predicting positive or negative rating
"""

#Split data into training and test sets with a 80/20 split for all binary models
#Based on the very low coefficients for both WordCount and vote, these variables were left out of the models.
X = PetSuppliesDF[['VaderCompound','Short','Verified','Long','IsImage']] #set independent variables for regression
Y = PetSuppliesDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run binary logistic regression
LR = linear_model.LogisticRegression()
LR.fit(X_train, Y_train)
print('Binary Logistic Intercept is:', LR.intercept_, '\n')
print('Binary Logistic Coefficients are:', LR.coef_, '\n')

#Look at ability of model to predict test set
#88.37% Accuracy
LRScore = round((LR.score(X_test, Y_test))*100,2)
print('Binary Logistic Model Score: ',LRScore,'%','\n')
Y_pred = LR.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Naive Bayes Classifier
NB = GaussianNB()
NB.fit(X_train, Y_train)

#Look at ability of model to predict test set
#87.69% Accuracy
NBScore = round((NB.score(X_test, Y_test))*100,2)
print('Naive Bayes Classifier Score is: ',NBScore,'%','\n')
Y_pred = NB.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Binary SVM
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

#Look at ability of model to predict test set
#88.24% Accuracy
SVMScore = round((svclassifier.score(X_test, Y_test))*100,2)
print('Binary SVM Score is: ',SVMScore,'%','\n')
Y_pred = svclassifier.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

"""
Second group of models are multiclass models for 1-5 rating
"""

#Split data into training and test sets with a 80/20 split for multiclass models
X = PetSuppliesDF[['VaderCompound','Short','Verified','Long','IsImage']] #set independent variables for regression
Y = PetSuppliesDF['overall'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run multinomial logistic regression
MLR = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=10000)
MLR.fit(X_train, Y_train)

#Look at ability of model to predict test set
#66.04% Accuracy
MLRScore = round((MLR.score(X_test, Y_test))*100,2)
print('Multinomial Logistic Model Score: ',MLRScore,'%','\n')
Y_pred = MLR.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run K Nearest Neighbors Algorithm
KNN = KNeighborsClassifier(n_neighbors = 15)
KNN.fit(X_train, Y_train)

#Look at ability of model to predict test set
#62.85% Accuracy
KNNScore = round((KNN.score(X_test, Y_test))*100,2)
print('K Nearest Neighbors Algorithm Model Score: ',KNNScore,'%','\n')
Y_pred = KNN.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Multiclass SVM
msvclassifier = SVC(kernel='linear')
msvclassifier.fit(X_train, Y_train)

#Look at ability of model to predict test set
#65.66% Accuracy
MSVMScore = round((msvclassifier.score(X_test, Y_test))*100,2)
print('Multiclass SVM Score is: ',MSVMScore,'%','\n')
Y_pred = msvclassifier.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Random Forest Algorithm
RF = RandomForestClassifier(n_estimators=5, random_state=0)
RF.fit(X_train, Y_train)

#Look at ability of model to predict test set
#63.27% Accuracy
RFScore = round((RF.score(X_test, Y_test))*100,2)
print('Random Forest Classifier Model Score: ',RFScore,'%','\n')
Y_pred = RF.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Implement TFIDF
tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1,5), analyzer='char')
X = tfidf.fit_transform(PetSuppliesDF['ReviewLemma'])
Y = PetSuppliesDF['overall']
X.shape, Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Implement Linear SVC model for TFIDF
#64.08% Accuracy
LSVC = LinearSVC(C = 10, class_weight='balanced', max_iter=10000)
LSVC.fit(X_train, Y_train)
LSVCScore = round((LSVC.score(X_test, Y_test))*100,2)
print('Linear SVC Model Score for TFIDF is: ',LSVCScore,'%', '\n')
Y_pred = LSVC.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Print out run times to decide how big of a data set to use
#Code run times: 1% 1 minute 43 seconds; 10% 1:08:57.
ElapsedSeconds = time.time() - StartTime
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)
print(convert(ElapsedSeconds))

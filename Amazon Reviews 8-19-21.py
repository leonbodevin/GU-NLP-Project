# -*- coding: utf-8 -*-
"""
Created on August 19, 2021

Amazonian Sentiments Capstone Project Group

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

#Record run time for code to decide on size of data set to use for analysis
StartTime = time.time()

#Reading in two very different data frames: Luxury Beauty and Video Games
LuxuryBeautyDF = pd.read_json('Luxury_Beauty.json', lines=True) # Read in json file as a dataframe
VideoGamesDF = pd.read_json('Video_Games_5.json', lines=True)
Blend = [LuxuryBeautyDF,VideoGamesDF]
BlendDF = pd.concat(Blend)
BlendDF = BlendDF.sample(frac = 0.1).reset_index() #To help with intense computational requirements, limiting to approximately 100,000 reviews

#Add column with Date from converted Unix time. Unfortunately results does not give time.
BlendDF["Date"] = pd.to_datetime(BlendDF["unixReviewTime"], unit='s')

#Create binary rating column: 0 (negative = 1-2), 1 (positive = 3-5)
#Binning decision came from running tests against validation data
conditions = [
    (BlendDF["overall"] > 2),
    (BlendDF["overall"] < 3)
    ]
values = [1, 0]
BlendDF['BinaryRating'] = np.select(conditions, values)

#Create column of review text with all lowercase, no punctuation, and no stopwords
nan_value = float("NaN") #Create na variable for blanks
BlendDF["reviewText"].replace("", nan_value, inplace=True) #Replace blanks with na variable
BlendDF.dropna(subset = ["reviewText"], inplace=True) #Drop all rows with na review text
BlendDF["ReviewNoFiller"] = BlendDF["reviewText"].str.replace('[^\w\s]','',regex=True) #Create column with review text with no punctuation
BlendDF["ReviewNoFiller"] = BlendDF["ReviewNoFiller"].str.lower() #Make all words lowercase
stopwords = stopwords.words('english') #Create stopwords variable
BlendDF["ReviewNoFiller"] = BlendDF["ReviewNoFiller"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) #Remove stop words
BlendDF["ReviewNoFiller"].replace("", nan_value, inplace=True,regex=True) #Replace blanks with na
BlendDF.dropna(subset = ["ReviewNoFiller"], inplace=True) #Drop all rows with na review text, reset indices

#Insert columns with tokenized review and summary
BlendDF["ReviewToken"] = BlendDF.apply(lambda row: word_tokenize(row["ReviewNoFiller"]), axis=1)

#Lemmatize all reviews and summaries, rejoin the strings
WNL = WordNetLemmatizer()
def lemmatize_text(text):
    return [WNL.lemmatize(w) for w in text]
BlendDF['ReviewToken'] = BlendDF.ReviewToken.apply(lemmatize_text)
BlendDF['ReviewLemma'] = BlendDF['ReviewToken'].apply(' '.join)

#Print out distribution of resulting review ratings
print(BlendDF['overall'].value_counts())

#Insert column with VADER sentiment analysis compound score of full review text, scale numbers from 1 to 5
vader = SentimentIntensityAnalyzer()
BlendDF["VaderCompound"] = [vader.polarity_scores(x)['compound'] for x in BlendDF["reviewText"]]
scaler = MinMaxScaler(feature_range=(1,5))
BlendDF["VaderCompound"] = scaler.fit_transform(BlendDF["VaderCompound"].values.reshape(-1,1))

#Insert column with review word count
BlendDF["WordCount"] = BlendDF["ReviewToken"].apply(len)

#What does word count distribution look like? Need visualization to decide how to bin data. Also look at descriptive statistics.
WordHist = BlendDF.hist(column = 'WordCount', bins=300)
plt.xlim([0,150])
print(BlendDF["WordCount"].describe()) #25% is 6 or less, 25% is 29 words or more, will bin accordingly

#Create column categorizing review word count as short (1) or not (0)
conditions = [
    (BlendDF["WordCount"] < 7),
    (BlendDF["WordCount"] > 6)
    ]
values = [1,0]
BlendDF['Short'] = np.select(conditions, values)

#Create column categorizing review word count as long (1) or not (0)
conditions = [
    (BlendDF["WordCount"] > 28),
    (BlendDF["WordCount"] < 29)
    ]
values = [1,0]
BlendDF['Long'] = np.select(conditions, values)

#Create column categorizing reviewer as verified (1) or not (0)
conditions = [
    (BlendDF['verified'] == True),
    (BlendDF['verified'] == False)
    ]
values = [1, 0]
BlendDF['Verified'] = np.select(conditions, values)

#Create binary column if the reviewer uploaded an image (1) or did not (0)
conditions = [
    (pd.notnull(BlendDF['image'])),
    (pd.isnull(BlendDF['image']))
    ]
values = [1, 0]
BlendDF['IsImage'] = np.select(conditions, values)

#Adjust vote column to allow for analysis
print(BlendDF.dtypes,'\n')
BlendDF['vote'] = BlendDF['vote'].str.replace('[^\w\s]','',regex=True) #Remove all punctuation from strings
BlendDF['vote'].replace('', '0', inplace=True) #Replace blanks with 0
BlendDF['vote'] = BlendDF['vote'].fillna('0') # Replace na values with 0
BlendDF['vote'] = BlendDF['vote'].astype({'vote': 'int32'})
print(BlendDF.dtypes,'\n')

#Read into five emotions lexicon, convert into five dictionaries
WarrinerWords = pd.read_csv("Warriner_BE.txt", sep='\t')
JoyDict = dict(zip(WarrinerWords.Word, WarrinerWords.Joy))
AngerDict = dict(zip(WarrinerWords.Word, WarrinerWords.Anger))
SadnessDict = dict(zip(WarrinerWords.Word, WarrinerWords.Sadness))
FearDict = dict(zip(WarrinerWords.Word, WarrinerWords.Fear))
DisgustDict = dict(zip(WarrinerWords.Word, WarrinerWords.Disgust))

#Create columns for five emotions, initialize all of them at zero
BlendDF['Joy'] = 0.0
BlendDF['Anger'] = 0.0
BlendDF['Sadness'] = 0.0
BlendDF['Fear'] = 0.0
BlendDF['Disgust'] = 0.0

#Iterate through all lists of words, assigning them scores for all emotions
for index, row in BlendDF.iterrows():
    for word in row['ReviewToken']:
        BlendDF.at[index,'Joy'] += JoyDict.get(word,0)
        BlendDF.at[index,'Anger'] += AngerDict.get(word,0)
        BlendDF.at[index,'Sadness'] += SadnessDict.get(word,0)
        BlendDF.at[index,'Fear'] += FearDict.get(word,0)
        BlendDF.at[index,'Disgust'] += DisgustDict.get(word,0)

#Create data frame for analysis
BlendDF = BlendDF[['ReviewLemma','VaderCompound','Short','Verified','Long','IsImage','WordCount','vote','Joy','Anger','Sadness','Fear','Disgust','overall','BinaryRating']]

# Write final dataframe into csv
BlendDF.to_csv(r'PetSupplies.csv', index = False)

#Print some of the dataframe to verify work
pd.set_option('display.max_columns', None) #So as not to truncate output
pd.set_option('display.max_rows', None) #So as not to truncate output
for col in BlendDF.columns: #Print column names
    print(col)
print(BlendDF.head()) # Print first five entries in dataframe
print("The number of rows in the data frame is:", len(BlendDF.index))

"""
TF-IDF Approach
"""

#Implement TF-IDF for binary model
tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1,5), analyzer='char')
X = tfidf.fit_transform(BlendDF['ReviewLemma'])
Y = BlendDF['BinaryRating']
X.shape, Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Implement Linear SVC model for binary TF-IDF
#89.55% Accuracy
LSVC = LinearSVC(C = 10, class_weight='balanced', max_iter=10000)
LSVC.fit(X_train, Y_train)
LSVCScore = round((LSVC.score(X_test, Y_test))*100,2)
print('Binary Linear SVC Model Score for TFIDF is:',LSVCScore,'%', '\n')
Y_pred = LSVC.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Implement TF-IDF for multiclass model
tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1,5), analyzer='char')
X = tfidf.fit_transform(BlendDF['ReviewLemma'])
Y = BlendDF['overall']
X.shape, Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Implement Linear SVC model for TF-IDF multiclass
#63.98% Accuracy
LSVC = LinearSVC(C = 10, class_weight='balanced', max_iter=10000)
LSVC.fit(X_train, Y_train)
LSVCScore = round((LSVC.score(X_test, Y_test))*100,2)
print('Multiclass Linear SVC Model Score for TFIDF is:',LSVCScore,'%', '\n')
Y_pred = LSVC.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

"""
VADER Score and other variables approach
First group of models are binary models predicting positive or negative rating
"""

#Split data into training and test sets with a 80/20 split for all binary models
#Based on the very low coefficients for both WordCount and vote, these variables were left out of the models.
X = BlendDF[['VaderCompound','Short','Verified','Long','IsImage']] #set independent variables for regression
Y = BlendDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run binary logistic regression
LR = linear_model.LogisticRegression()
LR.fit(X_train, Y_train)
print('Binary Logistic Intercept is:', LR.intercept_, '\n')
print('Binary Logistic Coefficients are:', LR.coef_, '\n')

#Look at ability of model to predict test set
#87.97% Accuracy
LRScore = round((LR.score(X_test, Y_test))*100,2)
print('Binary Logistic Model Score for VADER Score and other variables:',LRScore,'%','\n')
Y_pred = LR.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Binary SVM
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

#Look at ability of model to predict test set
#87.67% Accuracy
SVMScore = round((svclassifier.score(X_test, Y_test))*100,2)
print('Binary SVM Score for VADER Score and other variables:',SVMScore,'%','\n')
Y_pred = svclassifier.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Naive Bayes Classifier
NB = GaussianNB()
NB.fit(X_train, Y_train)

#Look at ability of model to predict test set
#87.28% Accuracy
NBScore = round((NB.score(X_test, Y_test))*100,2)
print('Binary Naive Bayes Classifier Score for VADER Score and other variables:',NBScore,'%','\n')
Y_pred = NB.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

"""
VADER Score and other variables approach
Second group of models are multiclass models for 1-5 rating
"""
#Split data into training and test sets with a 80/20 split for multiclass models
X = BlendDF[['VaderCompound','Short','Verified','Long','IsImage']] #set independent variables for regression
Y = BlendDF['overall'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run multinomial logistic regression
MLR = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=10000)
MLR.fit(X_train, Y_train)

#Look at ability of model to predict test set
#64.86% Accuracy
MLRScore = round((MLR.score(X_test, Y_test))*100,2)
print('Multinomial Logistic Model Score for VADER Score and other variables: ',MLRScore,'%','\n')
Y_pred = MLR.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Multiclass SVM
msvclassifier = SVC(kernel='linear')
msvclassifier.fit(X_train, Y_train)

#Look at ability of model to predict test set
#64.79% Accuracy
MSVMScore = round((msvclassifier.score(X_test, Y_test))*100,2)
print('Multiclass SVM Score is for VADER Score and other variables: ',MSVMScore,'%','\n')
Y_pred = msvclassifier.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run K Nearest Neighbors Algorithm
KNN = KNeighborsClassifier(n_neighbors = 15)
KNN.fit(X_train, Y_train)

#Look at ability of model to predict test set
#63.37% Accuracy
KNNScore = round((KNN.score(X_test, Y_test))*100,2)
print('K Nearest Neighbors Algorithm Model Score for VADER Score and other variables: ',KNNScore,'%','\n')
Y_pred = KNN.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Random Forest Algorithm
RF = RandomForestClassifier(n_estimators=5, random_state=0)
RF.fit(X_train, Y_train)

#Look at ability of model to predict test set
#60.43% Accuracy
RFScore = round((RF.score(X_test, Y_test))*100,2)
print('Random Forest Classifier Model Score for VADER Score and other variables: ',RFScore,'%','\n')
Y_pred = RF.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

"""
Five Emotions Approach
First group of models are binary models predicting positive or negative rating
SVM Models have been excluded due to high number of continuous variables makes processing power/time overwhelming
"""

#Split data into training and test sets with a 80/20 split for all binary models
#Based on the very low coefficients for both WordCount and vote, these variables were left out of the models.
X = BlendDF[['Joy','Anger','Sadness','Fear','Disgust']] #set independent variables for regression
Y = BlendDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run Naive Bayes Classifier
NB = GaussianNB()
NB.fit(X_train, Y_train)

#Look at ability of model to predict test set
#87.67% Accuracy
NBScore = round((NB.score(X_test, Y_test))*100,2)
print('Naive Bayes Classifier Score is for Five Emotions Model: ',NBScore,'%','\n')
Y_pred = NB.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run binary logistic regression
LR = linear_model.LogisticRegression()
LR.fit(X_train, Y_train)

#Look at ability of model to predict test set
#87.45 % Accuracy
LRScore = round((LR.score(X_test, Y_test))*100,2)
print('Binary Logistic Model Score for Five Emotions Model: ',LRScore,'%','\n')
Y_pred = LR.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

"""
Five Emotions Approach
Second group of models are multiclass models for 1-5 rating
SVM Models have been excluded due to high number of continuous variables makes processing power/time overwhelming
"""

#Split data into training and test sets with a 80/20 split for multiclass models
X = BlendDF[['Joy','Anger','Sadness','Fear','Disgust']] #set independent variables for regression
Y = BlendDF['overall'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run multinomial logistic regression
MLR = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=10000)
MLR.fit(X_train, Y_train)

#Look at ability of model to predict test set
#63.62% Accuracy
MLRScore = round((MLR.score(X_test, Y_test))*100,2)
print('Multinomial Logistic Model Score for Five Emotions Model: ',MLRScore,'%','\n')
Y_pred = MLR.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run K Nearest Neighbors Algorithm
KNN = KNeighborsClassifier(n_neighbors = 15)
KNN.fit(X_train, Y_train)

#Look at ability of model to predict test set
#62.55% Accuracy
KNNScore = round((KNN.score(X_test, Y_test))*100,2)
print('K Nearest Neighbors Algorithm Model Score for Five Emotions Model: ',KNNScore,'%','\n')
Y_pred = KNN.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

#Run Random Forest Algorithm
RF = RandomForestClassifier(n_estimators=5, random_state=0)
RF.fit(X_train, Y_train)

#Look at ability of model to predict test set
#55.39% Accuracy
RFScore = round((RF.score(X_test, Y_test))*100,2)
print('Random Forest Classifier Model Score for Five Emotions Model: ',RFScore,'%','\n')
Y_pred = RF.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0), '\n')

"""
Print out run times to decide how big of a data set to use
Code run times: 25 minutes 13 seconds for 10% of data
"""

ElapsedSeconds = time.time() - StartTime
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)
print(convert(ElapsedSeconds))

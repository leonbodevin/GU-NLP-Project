# -*- coding: utf-8 -*-
"""
Created on August 17, 2021

@author: Marcel Bodevin

Georgetown Data Science Certificate Program
"""

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

PetSuppliesDF = pd.read_json('Pet_Supplies_5.json', lines=True) # Read in json file as a dataframe
#PetSuppliesDF = PetSuppliesDF.sample(frac = 0.01) #For purposes of debugging code only: limit to 1% of dataframe. Use as needed.

#Add column with Date from converted Unix time, remove redundant columns. Unfortunately results does not give time.
PetSuppliesDF["Date"] = pd.to_datetime(PetSuppliesDF["unixReviewTime"], unit='s')
PetSuppliesDF = PetSuppliesDF.drop(['reviewTime', 'unixReviewTime'], axis=1)

#Create binary rating column: 0 (negative), 1 (positive)
conditions = [
    (PetSuppliesDF["overall"] > 3),
    (PetSuppliesDF["overall"] < 4)
    ]
values = [1, 0]
PetSuppliesDF['BinaryRating'] = np.select(conditions, values)

#Create column of review text with all lowercase, no punctuation, and no stopwords
nan_value = float("NaN") #Create na variable for blanks
PetSuppliesDF["reviewText"].replace("", nan_value, inplace=True) #Replace blanks with na variable
PetSuppliesDF.dropna(subset = ["reviewText"], inplace=True) #Drop all rows with na review text
PetSuppliesDF["ReviewNoFiller"] = PetSuppliesDF["reviewText"].str.replace('[^\w\s]','') #Create column with review text with no punctuation
PetSuppliesDF["ReviewNoFiller"] = PetSuppliesDF["ReviewNoFiller"].str.lower() #Make all words lowercase
stopwords = stopwords.words('english') #Create stopwords variable
PetSuppliesDF["ReviewNoFiller"] = PetSuppliesDF["ReviewNoFiller"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) #Remove stop words
PetSuppliesDF["ReviewNoFiller"].replace("", nan_value, inplace=True) #Replace blanks with na
PetSuppliesDF.dropna(subset = ["ReviewNoFiller"], inplace=True) #Drop all rows with na review text

#Create column of summary text with all lowercase, no punctuation, and no stopwords
PetSuppliesDF["SummaryNoFiller"] = PetSuppliesDF["summary"].str.replace('[^\w\s]','') #Create column with summary text with no punctuation
PetSuppliesDF["SummaryNoFiller"] = PetSuppliesDF["SummaryNoFiller"].str.lower() #Make column all lowercase
PetSuppliesDF["SummaryNoFiller"] = PetSuppliesDF["SummaryNoFiller"].fillna("") # Replace na values with blanks
PetSuppliesDF["SummaryNoFiller"] = PetSuppliesDF["SummaryNoFiller"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])) #Remove stop words

#Insert columns with tokenized review and summary
PetSuppliesDF["ReviewToken"] = PetSuppliesDF.apply(lambda row: word_tokenize(row["ReviewNoFiller"]), axis=1)
PetSuppliesDF["SummaryToken"] = PetSuppliesDF.apply(lambda row: word_tokenize(row["SummaryNoFiller"]), axis=1)

#Insert column with VADER sentiment analysis compound score of full review text
vader = SentimentIntensityAnalyzer()
PetSuppliesDF["VaderCompound"] = [vader.polarity_scores(x)['compound'] for x in PetSuppliesDF["reviewText"]]

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

#Create column categorizing reviewer as verified (1) or not (0), drop redundant column
conditions = [
    (PetSuppliesDF['verified'] == True),
    (PetSuppliesDF['verified'] == False)
    ]
values = [1, 0]
PetSuppliesDF['Verified'] = np.select(conditions, values)
PetSuppliesDF = PetSuppliesDF.drop(['verified'], axis=1)

#Create binary column if the reviewer uploaded an image (1) or did not (0), drop redundant column
conditions = [
    (pd.notnull(PetSuppliesDF['image'])),
    (pd.isnull(PetSuppliesDF['image']))
    ]
values = [1, 0]
PetSuppliesDF['IsImage'] = np.select(conditions, values)
PetSuppliesDF = PetSuppliesDF.drop(['image'], axis=1)

#Adjust vote column to allow for analysis
print(PetSuppliesDF.dtypes)
PetSuppliesDF['vote'] = PetSuppliesDF['vote'].str.replace('[^\w\s]','') #Remove all punctuation from strings
PetSuppliesDF['vote'].replace('', '0', inplace=True) #Replace blanks with 0
PetSuppliesDF['vote'] = PetSuppliesDF['vote'].fillna('0') # Replace na values with 0
PetSuppliesDF['vote'] = PetSuppliesDF['vote'].astype({'vote': 'int32'})
print(PetSuppliesDF.dtypes)

#Adjust data frame for analysis, dropping unneccesary columns
PetSuppliesDF = PetSuppliesDF[['VaderCompound','Short','Verified','IsImage','Long','overall','BinaryRating']]

# Write final dataframe into csv
PetSuppliesDF.to_csv(r'PetSupplies.csv', index = False)

#Print some of the dataframe to verify work
pd.set_option('display.max_columns', None) #So as not to truncate output
pd.set_option('display.max_rows', None) #So as not to truncate output
for col in PetSuppliesDF.columns: #Print column names
    print(col)
print(PetSuppliesDF.head()) # Print first five entries in dataframe

#Split data into training and test sets with a 80/20 split for Binary Logistic Regression
X = PetSuppliesDF[['VaderCompound']] #set independent variables for regression
Y = PetSuppliesDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run binary logistic regression
LR = linear_model.LogisticRegression()
LR.fit(X_train, Y_train)

#Look at ability of model to predict test set
#82.34 %
LRScore = round((LR.score(X_test, Y_test))*100,2)
print('Binary Logistic Model Score: ',LRScore,'%')

#Split data into training and test sets with a 80/20 split for Naive Bayes classifier
X = PetSuppliesDF[['VaderCompound']] #set independent variables for Bayes classifier
Y = PetSuppliesDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run Naive Bayes Classifier
NB = GaussianNB()
NB.fit(X_train, Y_train)

#Look at ability of model to predict test set
#82.38 %
NBScore = round((NB.score(X_test, Y_test))*100,2)
print('Naive Bayes Classifier Score is: ',NBScore,'%')

#Split data into training and test sets with a 80/20 split for Binary SVM
X = PetSuppliesDF[['VaderCompound']] #set independent variables for regression
Y = PetSuppliesDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run Binary SVM
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

#Look at ability of model to predict test set
#
SVMScore = round((svclassifier.score(X_test, Y_test))*100,2)
print('Binary SVM Score is: ',SVMScore,'%')

#Split data into training and test sets with a 80/20 split for multinomial logistics regression
X = PetSuppliesDF[['VaderCompound']] #set independent variables for regression
Y = PetSuppliesDF['overall'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run multinomial logistic regression
MLR = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
MLR.fit(X_train, Y_train)

#Look at ability of model to predict test set
#66.18 %
MLRScore = round((MLR.score(X_test, Y_test))*100,2)
print('Multinomial Logistic Model Score: ',MLRScore,'%')

#Split data into training and test sets with a 80/20 split for K Nearest Neighbors Algorithm
X = PetSuppliesDF[['VaderCompound']] #set independent variables for KNN
Y = PetSuppliesDF['overall'] #set dependent variable for KNN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run K Nearest Neighbors Algorithm
KNN = KNeighborsClassifier(n_neighbors = 15)
KNN.fit(X_train, Y_train)

#Look at ability of model to predict test set
#65.17 %
KNNScore = round((KNN.score(X_test, Y_test))*100,2)
print('K Nearest Neighbors Algorithm Model Score: ',KNNScore,'%')

#Split data into training and test sets with a 80/20 split for Multiclass SVM
X = PetSuppliesDF[['VaderCompound']] #set independent variables for regression
Y = PetSuppliesDF['overall'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run Multiclass SVM
msvclassifier = SVC(kernel='linear')
msvclassifier.fit(X_train, Y_train)

#Look at ability of model to predict test set
#
MSVMScore = round((msvclassifier.score(X_test, Y_test))*100,2)
print('Multiclass SVM Score is: ',MSVMScore,'%')

#Split data into training and test sets with a 80/20 split for Random Forest Algorithm
X = PetSuppliesDF[['VaderCompound']] #set independent variables for KNN
Y = PetSuppliesDF['overall'] #set dependent variable for KNN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets

#Run Random Forest Algorithm
RF = RandomForestRegressor(n_estimators=5, random_state=0)
RF.fit(X_train, Y_train)

#Look at ability of model to predict test set
#
RFScore = round((RF.score(X_test, Y_test))*100,2)
print('Random Forest Algorithm Model Score: ',RFScore,'%')


"""
Testing against validation data with both just VADER score vs. including all variables created above
Models using only VADER score were more accurate every single time, so final models will only include this variable

#Split data into training, validation, and test sets with a 60/20/20 split for binary logistics regression
#First use only Vader Compound Score as independent variable
X = PetSuppliesDF[['VaderCompound']] #set independent variables for regression
Y = PetSuppliesDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # Split off 20 validate set from training set

#Run binary logistic regression, print out intercept and coefficients
LR1 = linear_model.LogisticRegression()
LR1.fit(X_train, Y_train)
print('Binary Logistic Intercept for one variable: \n', LR1.intercept_)
print('Binary Logistic Coefficient for one variable: \n', LR1.coef_)

#Look at ability of model to predict validation set
#82.45% Accuracy
LR1Score = round((LR1.score(X_val, Y_val))*100,2)
print('Binary Logistic Model Score for one variable is: ',LR1Score,'%')

#Split data into training, validation, and test sets with a 60/20/20 split for binary logistics regression
#Next test multiple variables for independent variables
X = PetSuppliesDF[['VaderCompound','Short','Verified','IsImage','Long']] #set independent variables for regression
Y = PetSuppliesDF['BinaryRating'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # Split off 20 validate set from training set

#Run binary logistic regression, print out intercept and coefficients
LR2 = linear_model.LogisticRegression()
LR2.fit(X_train, Y_train)
print('Binary Logistic Intercept for multivarible: \n', LR2.intercept_)
print('Binary Logistic Coefficient for multivariable: \n', LR2.coef_)

#Look at ability of model to predict validation set
#82.05%
LR2Score = round((LR2.score(X_val, Y_val))*100,2)
print('Binary Logistic Model Score for multivariable is: ',LR2Score,'%')

#Split data into training, validation, and test sets with a 60/20/20 split for multinomial logistics regression
#First use only Vader Compound Score as independent variable
X = PetSuppliesDF[['VaderCompound']] #set independent variables for regression
Y = PetSuppliesDF['overall'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # Split off 20 validate set from training set

#Run multinomial logistic regression, print out intercept and coefficients
MLR1 = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
MLR1.fit(X_train, Y_train)
print('Multinomial Logistic Intercepts for single variable is: \n', MLR1.intercept_)
print('Multinomial Logistic Coefficients for single variable is: \n', MLR1.coef_)

#Look at ability of model to predict validation set
#66.42% Accuracy
MLR1Score = round((MLR1.score(X_val, Y_val))*100,2)
print('Multinomial Logistic Model Score for single variable is: ',MLR1Score,'%')

#Split data into training and test sets with a 80/20 split for multinomial logistics regression
#Next try multivariable for independent variables
X = PetSuppliesDF[['VaderCompound','Short','Verified','IsImage','Long']] #set independent variables for regression
Y = PetSuppliesDF['overall'] #set dependent variable for regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # Split off 20 validate set from training set

#Run multinomial logistic regression, print out intercept and coefficients
MLR2 = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
MLR2.fit(X_train, Y_train)
print('Multinomial Logistic Intercepts for multivariable: \n', MLR2.intercept_)
print('Multinomial Logistic Coefficients for multivariable: \n', MLR2.coef_)

#Look at ability of model to predict validation set
#66.23% Accuracy
MLR2Score = round((MLR2.score(X_val, Y_val))*100,2)
print('Multinomial Logistic Model Score for multivariable is: ',MLR2Score,'%')

#Split data into training and test sets with a 80/20 split for K Nearest Neighbors Algorithm
#First use only Vader Compound Score as dependent variable
X = PetSuppliesDF[['VaderCompound']] #set independent variables for KNN
Y = PetSuppliesDF['overall'] #set dependent variable for KNN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # Split off 20 validation set from training set

#Run multinomial logistic regression, print out intercept and coefficients
KNN1 = KNeighborsClassifier(n_neighbors = 15)
KNN1.fit(X_train, Y_train)

#Look at ability of model to predict validation set
#60.81% Accuracy
KNN1Score = round((KNN1.score(X_val, Y_val))*100,2)
print('K Nearest Neighbors Algorithm Model Score for single variable is: ',KNN1Score,'%')

#Split data into training and test sets with a 80/20 split for K Nearest Neighbors Algorithm
#Next try multivariable for independent variables
X = PetSuppliesDF[['VaderCompound','Short','Verified','IsImage','Long']] #set independent variables for KNN
Y = PetSuppliesDF['overall'] #set dependent variable for KNN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) #Split into 80/20 train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # Split off 20 validation set from training set

#Run multinomial logistic regression, print out intercept and coefficients
KNN2 = KNeighborsClassifier(n_neighbors = 15)
KNN2.fit(X_train, Y_train)

#Look at ability of model to predict validation set
#62.26%
KNN2Score = round((KNN2.score(X_val, Y_val))*100,2)
print('K Nearest Neighbors Algorithm Model Score for multivariable is: ',KNN2Score,'%')
"""
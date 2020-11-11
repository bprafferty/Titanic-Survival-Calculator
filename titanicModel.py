#data analysis and wrangling
import pandas as pd
import numpy as np

#machine learning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#saving the model
import pickle

#access data and create pandas dataframes
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
combine = [train_df, test_df]

#Wrangle the data

#Correcting
#start with correcting the data by dropping features that have no use
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

#Creating
#create new feature by extracting from existing
#pull title out of name feature
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#wrangle the new Title feature, compress every row with low counts into 1 row
#then compress variations into same row (Mlle == Miss, Ms == Miss, Mme == Mrs)
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#convert the categorical titles to ordinal for future use in the ML model
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#now drop Name and PassengerId feature, since they are no 
#longer needed in the training set
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

#convert the categorical feature Sex to numerical values 
#for future use in the ML model
sex_mapping = {'male': 0, 'female': 1}

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

#start with an empty array to contain the guessed ages
guess_ages = np.zeros((2,3))

#now iterate over Sex (0,1) and Pclass(1,2,3) to calc the guessed ages
for dataset in combine:
    for i in range(2):
        for j in range(3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            
            age_guess = guess_df.median()
            
            #convert random age float to nearest .5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
            

    for i in range(2):
        for j in range(3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
            
    dataset['Age'] = dataset['Age'].astype(int)

#create Age bands and determine correlations with Survived
train_df['AgeBand'] = pd.cut(train_df['Age'], 7)

#replace Age with ordinal values based upon these bands
for dataset in combine:
    dataset.loc[dataset['Age'] <= 11.429, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11.429) & (dataset['Age'] <= 22.857), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 22.857) & (dataset['Age'] <= 34.286), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 34.286) & (dataset['Age'] <= 45.714), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 45.714) & (dataset['Age'] <= 57.143), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 57.143) & (dataset['Age'] <= 68.571), 'Age'] = 5
    dataset.loc[dataset['Age'] > 68.571, 'Age'] = 6
    dataset['Age'] = dataset['Age'].astype(int)

#drop AgeBand, now that ages are placed in correct bands
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

#Creating
#create new feature called FamilySize which combines Parch and 
#SibSp, this will allow us to drop 2 columns and replace it with 1
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#Make a new feature called IsAlone to eliminate values of zero
for dataset in combine:
    dataset['IsAlone'] = 0
    #change value to 1 if family size is 1 person
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#drop Parch, SibSp, and FamilySize and keep IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

#create artificial feature combining Pclass and Age
for dataset in combine:
   dataset['Age*Class'] = dataset.Age * dataset.Pclass

#create artificial feature combining Title and Class
for dataset in combine:
   dataset['Title*Class'] = dataset.Title * dataset.Pclass

#Completing
#complete the embarked feature by find the mode, and 
#filling that value in all the null spots
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

#map Embarked to numeric values for the ML model
port_mapping = {'S': 0, 'C': 1, 'Q': 2}

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map(port_mapping)

#Completing
#fill the missing fare values by finding the mode and 
#replacing the nulls with it
freq_fare = train_df.Fare.dropna().mode()[0]

for dataset in combine:
    dataset['Fare'] = dataset['Fare'].fillna(freq_fare)

#create new feature called FareBand, just like AgeBand before
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

#convert Fare into ordinal values based upon results
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31.0, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

#remove FareBand feature
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

#Modeling
#time to train a model and predict a solution
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.5, random_state=1)
X_test = test_df.drop('PassengerId', axis=1).copy()

#preran gridsearchCV to find optimal parameters
classification_model = SVC(probability=True, C=100, gamma=0.01, kernel='rbf')
classification_model.fit(X_train, Y_train)

#Saving
#pickle the model to use in the app
pickle.dump(classification_model, open('./titanic_classifier.pkl', 'wb'))
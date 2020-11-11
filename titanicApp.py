import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
# Titanic Prediction App
This app predicts the **likelihood** that you would have survived the infamous disaster!
Data obtained from Kaggle: [Titanic: Machine Learning from Disaster](https://github.com/allisonhorst/palmerpenguins)
""")

st.sidebar.header('User Input Features')

title_mapping = {
                'Mr':1,
                'Miss':2,
                'Mrs':3,
                'Master':4,
                'Other':5 
}
sex_mapping = {
                'male':0,
                'female':1 
}
alone_mapping = {
                'No':0,
                'Yes':1
}
p_class_mapping = {
                'Upper Class':1,
                'Middle Class':2,
                'Lower Class':3
                
}

def user_input_features():
    t = st.sidebar.selectbox('Title',('Miss','Mr','Mrs','Master','Other'))
    title = title_mapping[t]

    s = st.sidebar.selectbox('Sex',('female','male'))
    sex = sex_mapping[s]

    ag = st.sidebar.slider('Age (Years)',1,100,21)
    if ag <= 11:
        age = 0
    elif ag > 11 and ag <= 22:
        age = 1
    elif ag > 22 and ag <= 34:
        age = 2
    elif ag > 34 and ag <= 45:
        age = 3
    elif ag > 45 and ag <= 57:
        age = 4
    elif ag > 57 and ag <= 68:
        age = 5
    else:
        age = 6

    al = st.sidebar.selectbox('Traveling alone?', ('Yes','No'))
    alone = alone_mapping[al]

    p = st.sidebar.selectbox('Socioeconomic Status',('Lower Class','Middle Class','Upper Class'))
    p_class = p_class_mapping[p]

    if p_class == 1:
        fare = 0
    elif p_class == 2 and age == 0:
        fare = 1
    elif p_class == 2 and age != 0:
        fare = 2
    else:
        fare = 3

    embarked = 0
    age_class = age * p_class
    title_class = title * p_class

    data = {
            #'PassengerID': '0',
            'Pclass': p_class,
            'Sex': sex,
            'Age': age,
            'Fare': fare,
            'Embarked': embarked,
            'Title': title,
            'IsAlone': alone,
            'Age*Class': age_class,
            'Title*Class': title_class
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

titanic_test = pd.read_csv('./input/test.csv')
df = pd.concat([input_df,titanic_test],axis=0)

#selects only the first row (the user input data)
df = df[:1]

#reads in saved classification model
load_svm = pickle.load(open('titanic_classifier.pkl', 'rb'))

#apply model to make predictions
prediction = load_svm.predict(input_df)
prediction_probability = load_svm.predict_proba(input_df)


st.subheader('Prediction')
result = np.array(['You did not survive.','You Survived!'])
st.write(result[prediction][0])

st.subheader('Prediction Probability')
st.write('Survival Likelihood: {0:.2f}%'.format(prediction_probability[0][1] * 100))
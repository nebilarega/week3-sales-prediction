import streamlit as st
import pandas as pd
import pickle

from config import features, salesDependingFeatures, model_load

# feature update


def feature_update(li, features):
    final_features = []
    for i in li:
        for j in features:
            if(i == j):
                final_features.append(features.index(j))
    return final_features

# page functioning


def app():
    html1 = '''
                <style>
                #heading{
                  color: #E65142;
                  text-align:top-left;
                  font-size: 45px;
                }
                </style>
                <h1 id = "heading"> Sales Data Prediction</h1>
            '''
    st.markdown(html1, unsafe_allow_html=True)

    li = st.multiselect(
        "Select the feature/features whose value can be manually updated ", features)
    list = feature_update(li, features)
    value = []
    for i in list:
        if i in ['StateHoliday', 'StoreType', 'Assortment']:
            string_val = st.text_input("Enter the values " + features[i])
            value.append(string_val)
        else:
            number = st.number_input("Enter the values " + features[i])
            value.append(number)

    for i in range(len(list)):
        salesDependingFeatures[list[i]] = value[i]
    d = {"Feature ": features, "Value for Prediction": salesDependingFeatures}
    st.subheader("Default values")
    test_data = pd.DataFrame([salesDependingFeatures], columns=features)
    st.write(test_data)
    loaded_model = pickle.load(
        open('models/2022_09_10-01_58_24_PM_randomforest.pkl', 'rb'))

    model_ = loaded_model.predict(test_data)
    st.subheader("The Predicted Value ")
    st.write(model_)

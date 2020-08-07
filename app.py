import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Schema Matching Web App")
    st.markdown("Please upload your spreadsheet in the sidebar!")
    st.sidebar.title("Schema Matching Spreadsheet Upload")
    st.sidebar.markdown("Please upload below!")

    #uploads the csv file
    st.sidebar.subheader("Upload a File")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    #Here are the two forms of text we can use
    st.subheader("How does this work?")
    st.text("Here is a paragraph of text that explains what this service does.")

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv'])

    '''
    if uploaded_file != None:
        if st.sidebar.checkbox("Show raw data ",False):
            st.subheader("Provider Data Set (Classification)")
            data = pd.read_csv(uploaded_file)
            st.write(data)
    '''

    if st.sidebar.button("Classify", key='classify'):
        #make this a dataframe - convert the uploaded file to csv to a dataframe
        output = evaluate(uploaded_file)
        st.write(output)

def evaluate(uploaded_file):
    test_data = pd.read_csv(uploaded_file)
    test_data = test_data.dropna(thresh=len(test_data.index)/2, axis='columns')
            # pre-processing input data, remove columns with empty name
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
    test_data = test_data.astype(str)   # convert all labels to str, mod 7/8rip()

            # pre-processing input data, remove columns with empty name
    test_data = test_data[test_data.columns.dropna()]  # drop col w/o no header
    test_data.columns = test_data.columns.str.strip()     # remove space b/a header

    # capitalize all the labels
    for i in range(len(test_data.iloc[-1,:])):
        s = test_data.iloc[-1,i]
        lst = [word[0].upper() + word[1:] for word in s.split()]
        test_data.iloc[-1,i] = " ".join(lst)
            # extract the label, which is the last row
    label = test_data.tail(1).to_dict('records')[0]

            # drop the label if it is exists. otherwise drop the last row
    test_data = test_data.drop(test_data.tail(1).index)
            # test_data = test_data.astype(str)
            # test_data = test_data.fillna("NA")
            # test_data.columns = test_data.columns.str.strip()

    loaded_model = pickle.load(open('matcher1.model', 'rb'))
    predicted_output = loaded_model.make_prediction(test_data)
    predicted_mapping = {}
    for k,v in predicted_output.items():
        predicted_mapping[k] = v[0]

    json_output = json.dumps(predicted_output, indent=4)
    return json_output


if __name__ == '__main__':
    main()

import streamlit as st
from src.predict import predict_message
st.title('Spam Email Classifier')
st.write('Please write a message below to check if it looks like spam.')

user_input = st.text_area('Message text')

if st.button('Check'):
    if user_input.strip() == '':
        st.warning('Please enter a message first.')
    else:
        result = predict_message(user_input)
        if result == 'Spam':
            st.error(f'Prediction: {result}')
        else:
            st.success(f'Prediction: {result}')
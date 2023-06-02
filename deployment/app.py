import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page: ', ('Explore', 'Prediction'))

if navigation == 'Explore':
    eda.run()
else:
    prediction.run()

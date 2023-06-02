import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from tensorflow.keras.models import load_model


# load files
with open('full_pipeline.pkl', 'rb') as file_1:
    full_pipeline = pickle.load(file_1)

model_ann = load_model('best_model.h5')


def run():
    with st.form(key='from_churn'):
        st.title('Prediction Page')
        st.subheader('We Calculate your metric to check Customer Churn Score')
        st.write('*`Please fill columns below to predict`*')

        # buat variabel
        age = st.number_input('Age', min_value=0,
                              max_value=99, value=40, step=1)

        st.write('F = Female | M = Male')
        gender_pick = st.radio('Gender', ('M', 'F'), index=1)

        region = st.selectbox('Region', ('City', 'Village', 'Town'))

        member = st.selectbox('Membership', ('No Membership', 'Basic Membership', 'Silver Membership',
                                             'Premium Membership', 'Gold Membership', 'Platinum Membership'))

        joined = st.date_input('Date when a customer became member',
                               datetime.date(2023, 12, 31))

        join_ref = st.radio(
            'Are Customer Joined Using Referral?', ('Yes', 'No'), index=1)

        pref_offer = st.selectbox('Type of offer to customer', ('Without Offers', 'Credit/Debit Card Offers',
                                                                'Gift Vouchers/Coupons'))

        medium = st.selectbox(
            'Type of Device', ('Smartphone', 'Desktop', 'Both'))

        internet_opt = st.selectbox(
            'Type of internet service', ('Wi-Fi', 'Fiber_Optic', 'Mobile_Data'))

        last_visit = st.time_input('The last time a customer visited the website',
                                   datetime.time(15, 00), step=300)

        last_login = st.number_input('Number of days since a customer last logged into the website',
                                     min_value=0, max_value=366, value=4, step=1)

        time_spent = st.number_input('Average time (minutes) customer spent in the platform',
                                     min_value=0, max_value=5000, value=30, step=1)

        trx_value = st.number_input('Average transaction value of a customer', min_value=0,
                                    max_value=10_000, value=500, step=5)

        login_days = st.number_input('Number of times a customer has logged in to the website', min_value=0,
                                     max_value=10_000, value=25, step=1)

        point_wallet = st.number_input('Points awarded to a customer on each transaction', min_value=0,
                                       max_value=50_000, value=1000, step=10)

        used_discount = st.radio(
            'Whether a customer uses special discounts offered', ('Yes', 'No'), index=1)

        pref = st.radio('Whether a customer prefers offers',
                        ('Yes', 'No'), index=1)

        past_comp = st.radio(
            'Whether a customer has raised any complaints', ('Yes', 'No'), index=1)

        comp_status = st.selectbox('Whether the complaints raised by a customer was resolved', ('No Information Available', 'Not Applicable', 'Unsolved',
                                                                                                'Solved', 'Solved in Follow-up'))

        feedback = st.selectbox('Feedback provided by a customer', ('Poor Website',
                                                                    'Poor Customer Service', 'Too many ads',
                                                                    'Poor Product Quality', 'No reason specified',
                                                                    'Products always in Stock', 'Reasonable Price',
                                                                    'Quality Customer Care', 'User Friendly Website'))

        st.markdown('---')
        submitted = st.form_submit_button('Predict')

        data_inf = {
            'age': age,
            'gender': gender_pick,
            'region_category': region,
            'membership_category': member,
            'joined_through_referral': join_ref,
            'preferred_offer_types': pref_offer,
            'medium_of_operation': medium,
            'internet_option': internet_opt,
            'days_since_last_login': last_login,
            'avg_time_spent': time_spent,
            'avg_transaction_value': trx_value,
            'avg_frequency_login_days': login_days,
            'points_in_wallet': point_wallet,
            'used_special_discount': used_discount,
            'offer_application_preference': pref,
            'past_complaint': past_comp,
            'complaint_status': comp_status,
            'feedback': feedback
        }

        data_inf = pd.DataFrame([data_inf])
        # st.dataframe(data_inf.T, width=800, height=495)

        if submitted:
            # predict using full_pipeline
            data_pipeline = full_pipeline.transform(data_inf)
            y_pred_inf = model_ann.predict(data_pipeline)
            y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)
            y_pred_inf

            if y_pred_inf == 0:
                st.write('Not Churn')
            else:
                st.write('Churn')

        st.markdown('---')

    if __name__ == '__main__':
        run()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title='Churn Predictor',
    layout='wide',
    initial_sidebar_state='expanded'
)


def run():
    # judul
    st.title('**Churn Exploration**')
    st.subheader('Explore The Churn Dataset')

    # tambah gambar
    image = Image.open('churn.jpg')
    st.image(image)
    st.markdown('---')

    markdown_text = '''
    ## Background
    In today's competitive business landscape, customer churn has become a 
    significant concern for many companies. Customer churn refers to the 
    phenomenon where customers discontinue using a company's products or 
    services. Churn can have a negative impact on a company's revenue, 
    growth, and overall success. Therefore, companies are increasingly 
    focused on identifying customers who are likely to churn so that they 
    can take proactive measures to retain them.
    
    ## Objective
    The objective of this project is to develop a deep learning model for 
    churn prediction. The company wants to minimize the risk of customer 
    churn by accurately predicting which customers are likely to stop using 
    their products or services. By identifying potential churners in advance,
    the company can take targeted actions and implement retention strategies 
    to reduce churn rates and maximize customer loyalty.
    
    ## About Dataset
    
    |         Variable        |                                         Description                                           |
    |-------------------------|-----------------------------------------------------------------------------------------------|
    | user_id                 | ID of a customer                                                                              |                        
    | age                     | Age of a customer                                                                             |
    | gender                  | Gender of a customer                                                                          |
    | region category         | Region that a customer belongs to                                                             |
    | membership category     | Category of the membership that a customer is using                                           |
    | joining date            | Date when a customer became a member                                                          |
    | joined through referal  | Whether a customer joined using any referral code or ID                                       |
    | preferred_offer types   | Type of offer that a customer prefers                                                         |
    | medium_of operation     | Medium of operation that a customer uses for transactions                                     |
    | internet option         | Type of internet service a customer uses                                                      |
    | last visit time         | The last time a customer visited the website                                                  |
    | days since last login   | Number of days since a customer last logged into the website                                  |
    | average time spent      | Average time spent by a customer on the website                                               |
    | average transaction     | Average transaction value of a customer                                                       |
    | average freq login days | Number of times a customer has logged in to the website                                       |
    | point in wallet         | Points awarded to a customer on each transaction                                              |
    | used spesial discount   | Whether a customer uses special discounts offered                                             |
    | offer app preference    | Whether a customer prefers offers                                                             |
    | past complaint          | Whether a customer has raised any complaints                                                  |
    | complaint status        | Whether the complaints raised by a customer was resolved                                      |
    | feedback                | Feedback provided by a customer                                                               |
    | churn risk score        | Churn Score                                                                                   |
    
    '''
    st.markdown(markdown_text)
    st.markdown('---')

    st.subheader('**Data Exploratory**')
    st.markdown('---')

    st.write('### Customer Information')

    # show dataset
    data = pd.read_csv('churn.csv')
    st.dataframe(data)
    st.markdown('---')

    st.write("### Today's Condition")
    st.markdown('---')

    # show distribusi customer churn
    fig, ax = plt.subplots()
    plt.pie(data['churn_risk_score'].value_counts(),
            labels=['Churn', 'Not-Churn'],
            autopct='%1.1f%%',
            colors=['Grey', 'Orange'],
            startangle=40,
            explode=[0.05, 0])
    plt.title('Customer Churn Percentage')
    plt.axis('equal')
    st.pyplot(fig)

    '''
    Based on the above graph, it can be observed that the distribution of Churn Risk 
    Score ***tends to be evenly divided*** among its values, indicating that the feature 
    follows a normal distribution
    '''

    # visual numerical
    st.subheader('Chart Based on Metrics')
    st.markdown('---')

    choice = st.selectbox('Pick Numeric Columns: ', ('age', 'days_since_last_login', 'avg_time_spent',
                                                     'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=data, x=choice, fill=True,
                hue='churn_risk_score', palette='inferno')
    ax.set_title(choice.capitalize()+' Ratio')
    st.pyplot(fig)
    st.markdown('---')

    # visual categorical
    choice_2 = st.selectbox('Pick Category Column : ', ('gender', 'region_category',
                            'membership_category', 'joined_through_referral', 'preferred_offer_types',
                                                        'medium_of_operation', 'internet_option', 'used_special_discount',
                                                        'offer_application_preference', 'past_complaint', 'complaint_status', 'feedback'))
    fig = plt.figure(figsize=(15, 10))
    sns.countplot(data=data, x=choice_2,
                  hue='churn_risk_score', palette='viridis')
    plt.xlabel(choice_2.capitalize())
    plt.ylabel('Count')
    plt.title(choice_2.capitalize()+' Ratio')
    plt.legend(title='Churn Risk Score')
    st.pyplot(fig)


if __name__ == '__main__':
    run()

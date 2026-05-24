# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import main
import joblib
from datetime import datetime

# ---------------------------------------------------------------------------------------

def get_user_input(data):

    st.sidebar.write(f"**Order Related Information**")
    date = st.sidebar.date_input("What is the order date?")
    time = st.sidebar.time_input("What is the time of order?",step=60)

    picked_time = st.sidebar.time_input("When you picked the order?",step=60)

    type_of_order = st.sidebar.selectbox("What is the type of your order?",data['Type_of_order'].unique())

    multiple_deliveries = st.sidebar.selectbox("How many deliveries are combined?",
                                               data['multiple_deliveries'].unique())

    st.sidebar.write(f"**Location Related Information**")

    restaurant_latitude = st.sidebar.text_input("What is the restaurant latitude?", "14.829222")
    restaurant_longitude = st.sidebar.text_input("What is the restaurant longitude?", "67.920922")
    delivery_location_latitude = st.sidebar.text_input("What is the delivery location latitude?", "14.929222")
    delivery_location_longitude = st.sidebar.text_input("What is the delivery location longitude?", "68.860922")

    st.sidebar.write(f"**Delivery person related Information**")


    age = st.sidebar.slider("What is the age of delivery person?",
                                            int(data['Delivery_person_Age'].min()),
                                            int(data['Delivery_person_Age'].max()),
                                            int(data['Delivery_person_Age'].mean()))

    ratings = st.sidebar.slider("What is tthe ratings given by delivery person?",
                  data["Delivery_person_Ratings"].min(),
                  data['Delivery_person_Ratings'].max(),
                  data['Delivery_person_Ratings'].mean())

    vehicle_condition = st.sidebar.selectbox("What is the condition of Vehicle?",
                                         data['Vehicle_condition'].unique())

    type_of_vehicle = st.sidebar.selectbox("which type of vehicle do you have?",
                                       data['Type_of_vehicle'].unique())



    st.sidebar.write(f"**City related Information**")

    city = st.sidebar.selectbox("What is the type of City?",data['City'].unique())

    st.sidebar.write(f"**Weather related/Event related Information**")

    weather = st.sidebar.selectbox("How is the weather?",
                     ['Fog','Stormy','Cloudy','Sandstorms','Windy','Sunny'])

    festival = st.sidebar.selectbox("Is there any festival?",data['Festival'].unique())

    road_density = st.sidebar.selectbox("What is road traffic density?",
                                    data['Road_traffic_density'].unique())



    X = pd.DataFrame({'Delivery_person_Age':age,
                    'Delivery_person_Ratings':ratings,
                    'Restaurant_latitude':restaurant_latitude,
                    'Restaurant_longitude':restaurant_longitude,
                    'Delivery_location_latitude':delivery_location_latitude,
                    'Delivery_location_longitude':delivery_location_longitude,
                    'Order_Date':str(date),
                    'Time_Orderd':str(time),
                    'Time_Order_picked':str(picked_time),
                    'Weatherconditions':weather,
                    'Road_traffic_density':road_density,
                    'Vehicle_condition':vehicle_condition,
                    'Type_of_order':type_of_order,
                    'Type_of_vehicle':type_of_vehicle,
                    'multiple_deliveries':multiple_deliveries,
                    'Festival':festival,
                    'City':city},index = [0]
                      )
    return X

# ---------------------------------------------------------------------------------------------


st.set_page_config(page_title="Food Delivery Time Prediction", page_icon=None, layout="centered",
                       initial_sidebar_state="auto")

st.title("Food Delivery Time Prediction")
st.divider()

data = pd.read_csv("Data/train.csv")

cleaned_data = main.Cleaning_steps.fit_transform(data)

st.image("img/image.jpeg",width=700)

st.write(""" The food delivery time prediction model is vital in ensuring prompt and accurate delivery in the food delivery industry. Leveraging advanced data cleaning techniques and feature engineering, a robust food delivery time prediction model is developed.
            This model predicts food delivery time based on a range of factors, including order details, location, city, delivery person information, and weather conditions.  
             """)

st.sidebar.header("User input parameters")

input_df = get_user_input(cleaned_data)

model = joblib.load("Code/model1.joblib")

st.subheader("Order Details")

order_time = datetime.strptime(input_df['Time_Orderd'].iloc[0],"%H:%M:%S").strftime("%I:%M %p")
st.write(f"**Order was placed on** {order_time}")

picked_time = datetime.strptime(input_df['Time_Order_picked'].iloc[0],"%H:%M:%S").strftime("%I:%M %p")
st.write(f"**Order was picked up at** {picked_time}")

st.subheader('Prediction')


prediction = model.predict(input_df)[0]

st.write(f"**Order is deliverd in** {prediction:.1f} **minutes**")
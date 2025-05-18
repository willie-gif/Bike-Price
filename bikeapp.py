import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import firebase_admin
#import requests
#import os
#import time
#import io
#from datetime import datetime
#from firebase_admin import credentials, auth, initialize_app
#from firebase_admin import firestore
from streamlit_option_menu import option_menu


# Set up the page configuration
st.set_page_config(page_title="Bike Prices", layout="wide",page_icon="🚴")

# Load the trained model
def load_model():
    with open(r"C:\Users\User\bike.pkl","rb") as file:
        data=pickle.load(file)
    return data

data= load_model()



model = data["regressor"]
month_le = data["month_le"]
country_le = data["country_le"]
state_le = data["state_le"]
category_le = data["category_le"]
sub_category_le = data["sub_category_le"]
product_le = data["product_le"]
scaler = data["scaler"]



# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Bike Price Prediction System",
        ["Home", "Details"],
        icons=["house", "","gear", "box-arrow-right"],
        menu_icon="🚴",
        default_index=0
        )
    



# HOME SECTION
if selected == "Home":
    st.title("🏠 Home")
    st.write("Welcome to the Bike Price Prediction System.")

# DETAILS SECTION WITH SUBMENU
elif selected == "Details":
    st.title("Customer & Product Details")
    st.write("Please Fill Out The Following Details:")
        
# User input columns
    col1, col2 = st.columns(2)

    # Input options
    state_options = ('British Columbia', 'New South Wales', 'Victoria', 'Oregon',
       'California', 'Saarland', 'Seine Saint Denis', 'Moselle',
       'Queensland', 'England', 'Nord', 'Washington', 'Hessen',
       'Nordrhein-Westfalen', 'Hamburg', 'Loir et Cher', 'Kentucky',
       'Seine (Paris)', 'South Australia', 'Loiret', 'Alberta', 'Bayern',
       'Hauts de Seine', 'Yveline', 'Essonne', "Val d'Oise", 'Tasmania',
       'Seine et Marne', 'Val de Marne', 'Pas de Calais',
       'Charente-Maritime', 'Garonne (Haute)', 'Brandenburg', 'Texas',
       'New York', 'Florida', 'Somme', 'Illinois', 'South Carolina',
       'North Carolina', 'Georgia', 'Virginia', 'Ohio', 'Ontario',
       'Wyoming', 'Missouri', 'Montana', 'Utah', 'Minnesota',
       'Mississippi', 'Massachusetts', 'Arizona', 'Alabama')
    country_options = ('Canada', 'Australia', 'United States', 'Germany', 'France',
       'United Kingdom', 'Austria', 'Denmark', 'Slovakia', 'Estonia',
       'Bulgaria', 'Malta', 'Lithuania', 'Finland', 'Poland', 'Italy',
       'Netherlands', 'Romania', 'Slovenia', 'Ireland', 'Luxembourg',
       'Portugal', 'Latvia', 'Sweden', 'Spain', 'Cyprus', 'Croatia',
       'Greece', 'Belgium', 'Czech Republic', 'Hungary')
    month_options = ('November', 'March', 'May', 'February', 'July', 'August','September', 'January', 'December', 'June', 'October', 'April')
    Category_options = ("Accessories","Clothing","Bikes")
    sub_category_options = ("Tires & Tubes","Bike Racks","Bottles & Cages","Caps","Fenders","Cleaners","Touring Bikes","Gloves","Helmets","Jerseys","Hydration Packs","Mountain Bikes","Road Bikes","Socks","Vests","Bike Stands")
    product_options = ('Hitch Rack - 4-Bike', 'All-Purpose Bike Stand',
        'Mountain Bottle Cage', 'Water Bottle - 30 oz.',
        'Road Bottle Cage', 'AWC Logo Cap', 'Bike Wash - Dissolver',
        'Fender Set - Mountain', 'Half-Finger Gloves, L',
        'Half-Finger Gloves, M', 'Half-Finger Gloves, S',
        'Sport-100 Helmet, Black', 'Sport-100 Helmet, Red',
        'Sport-100 Helmet, Blue', 'Hydration Pack - 70 oz.',
        'Short-Sleeve Classic Jersey, XL',
        'Short-Sleeve Classic Jersey, L', 'Short-Sleeve Classic Jersey, M',
        'Short-Sleeve Classic Jersey, S', 'Long-Sleeve Logo Jersey, M',
        'Long-Sleeve Logo Jersey, XL', 'Long-Sleeve Logo Jersey, L',
        'Long-Sleeve Logo Jersey, S', 'Mountain-100 Silver, 38',
        'Mountain-100 Silver, 44', 'Mountain-100 Black, 48',
        'Mountain-100 Silver, 48', 'Mountain-100 Black, 38',
        'Mountain-200 Silver, 38', 'Mountain-100 Black, 44',
        'Mountain-100 Silver, 42', 'Mountain-200 Black, 46',
        'Mountain-200 Silver, 42', 'Mountain-200 Silver, 46',
        'Mountain-200 Black, 38', 'Mountain-100 Black, 42',
        'Mountain-200 Black, 42', 'Mountain-400-W Silver, 46',
        'Mountain-500 Silver, 40', 'Mountain-500 Silver, 44',
        'Mountain-500 Black, 48', 'Mountain-500 Black, 40',
        'Mountain-400-W Silver, 42', 'Mountain-500 Silver, 52',
        'Mountain-500 Black, 52', 'Mountain-500 Silver, 42',
        'Mountain-500 Black, 44', 'Mountain-500 Silver, 48',
        'Mountain-400-W Silver, 38', 'Mountain-400-W Silver, 40',
        'Mountain-500 Black, 42', 'Road-150 Red, 48', 'Road-150 Red, 62',
        'Road-750 Black, 48', 'Road-750 Black, 58', 'Road-750 Black, 52',
        'Road-150 Red, 52', 'Road-150 Red, 44', 'Road-150 Red, 56',
        'Road-750 Black, 44', 'Road-350-W Yellow, 40',
        'Road-350-W Yellow, 42', 'Road-250 Black, 44',
        'Road-250 Black, 48', 'Road-350-W Yellow, 48',
        'Road-550-W Yellow, 44', 'Road-550-W Yellow, 38',
        'Road-250 Black, 52', 'Road-550-W Yellow, 48', 'Road-250 Red, 58',
        'Road-250 Black, 58', 'Road-250 Red, 52', 'Road-250 Red, 48',
        'Road-250 Red, 44', 'Road-550-W Yellow, 42',
        'Road-550-W Yellow, 40', 'Road-650 Red, 48', 'Road-650 Red, 60',
        'Road-650 Black, 48', 'Road-350-W Yellow, 44', 'Road-650 Red, 52',
        'Road-650 Black, 44', 'Road-650 Red, 62', 'Road-650 Red, 58',
        'Road-650 Black, 60', 'Road-650 Black, 58', 'Road-650 Black, 52',
        'Road-650 Black, 62', 'Road-650 Red, 44',
        "Women's Mountain Shorts, M", "Women's Mountain Shorts, S",
        "Women's Mountain Shorts, L", 'Racing Socks, L', 'Racing Socks, M',
        'Mountain Tire Tube', 'Touring Tire Tube', 'Patch Kit/8 Patches',
        'HL Mountain Tire', 'LL Mountain Tire', 'Road Tire Tube',
        'LL Road Tire', 'Touring Tire', 'ML Mountain Tire', 'HL Road Tire',
        'ML Road Tire', 'Touring-1000 Yellow, 50', 'Touring-1000 Blue, 46',
        'Touring-1000 Yellow, 60', 'Touring-1000 Blue, 50',
        'Touring-3000 Yellow, 50', 'Touring-3000 Blue, 54',
        'Touring-3000 Blue, 58', 'Touring-3000 Yellow, 44',
        'Touring-3000 Yellow, 54', 'Touring-3000 Blue, 62',
        'Touring-3000 Blue, 44', 'Touring-1000 Blue, 54',
        'Touring-1000 Yellow, 46', 'Touring-1000 Blue, 60',
        'Touring-3000 Yellow, 62', 'Touring-1000 Yellow, 54',
        'Touring-2000 Blue, 54', 'Touring-3000 Blue, 50',
        'Touring-3000 Yellow, 58', 'Touring-2000 Blue, 46',
        'Touring-2000 Blue, 50', 'Touring-2000 Blue, 60',
        'Classic Vest, L', 'Classic Vest, M', 'Classic Vest, S')

    # Collecting inputs
    with col1:
        country = st.selectbox("Select Country", country_options)
    with col2:
        month = st.selectbox("Select Month", month_options)
    with col1:
        Age = st.slider("Age ", 5, 90, 5)
    with col2:
        state = st.selectbox("Select state", state_options)    
    with col1:
        Category = st.selectbox("Category", Category_options)
    with col2:
        Quantity = st.slider("Quantity", 1, 100, 1)
    with col1:
        sub_category = st.selectbox("Sub_Category", sub_category_options)
    with col2:
        products = st.selectbox("Product", product_options)


    

        # Predict Button
    if st.button("Predict"):
        x_new=np.array([[month,Age,country,state,Category,sub_category,products,Quantity]])
                # Encode new data
        x_new[:, 0] = month_le.transform(x_new[:, 0])
        x_new[:, 2] = country_le.transform(x_new[:, 2])
        x_new[:, 3] = state_le.transform(x_new[:, 3])
        x_new[:, 4] = category_le.transform(x_new[:, 4])
        try:
            x_new[:, 5] = sub_category_le.transform(x_new[:, 5])
        except ValueError:
            st.error(f"Sub-category '{x_new[:, 5][0]}' not recognized. Please choose a valid option.")
            st.stop()

        x_new[:, 6] = product_le.transform(x_new[:, 6])

        x_new = x_new.astype(float)
        x_new = scaler.transform(x_new)
        

        bike=model.predict(x_new)
        st.subheader(f"The Bike Price is ${bike[0]}")

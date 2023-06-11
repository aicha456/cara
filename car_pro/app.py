import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import joblib
import os
import hashlib
import requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie
from manage import *
import statsmodels.api as sm


df = pd.read_csv("car_pro/car (2).csv")

@st.cache(allow_output_mutation=True)
def model_loader(path):
    mode = joblib.load(path)
    return mode

moodel = model_loader("car_pro/tree.pkl" )



def best_model(bran):
    a = df[df['brand'] == bran]
    o = pd.pivot_table(a, index='model', values='price',
                       aggfunc=np.mean).sort_values(by='price', ascending=False)
    v = o.head(4)
    ax = v.T.plot(kind='bar', ylabel='price')
    plt.title(bran + ' vs Price')
    plt.ylabel('Price')
    plt.xlabel(bran)
    st.pyplot(plt)

    return
def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

def find_model(brand):
    mo = df[df['brand'] == brand]
    model = mo['model'].unique()  # return series of filter model name for specific brand.
    return list(model)  # return list of filter model name for specific brand.


def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False


def main():
    st.title("cars price preadiction ")

    menu = ["Home", "Login", "Signup"]
    sub_menu = ["Plot", "Prediction", "Metrics"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        lottie_he = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_EPex2ugpSG.json")
        st_lottie(
            lottie_he,
        )
        st.header('🚙 Little Info About the Project 🚙')
        st.markdown("""
  
                   ###### this app analysis and  predict used car 🚙 price by giving some information like car model , how much the car has been driven and so on.\n
                  
                     
                     <ul>
                      <li>UK Used Car Data preliminary analysis </li>
                      <li> Checking how many models are most populars</li>
                     <li>Checking the average price for each  model</li>
                     
                     <li>Checking the average price for each  model</li>
                     
                    </ul>
                     
                     
                   

                    I am sharing the full project's notebooks along with dataset. \n
                    
                    In case want to contact with me -  aichaamisaid12@gmail.com📫
        """, unsafe_allow_html=True)

        st.header("""Untll then ❤""")







    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')





        if st.sidebar.checkbox('Login'):
            create_usertabel()
            has_h = generate_hashes(password)
            result = login_user(username, verify_hashes(password, has_h))

            if result:
                st.success("welcome  {}  ".format(username))

                activity = st.selectbox("activity", sub_menu)
                if activity == 'Plot':
                    st.subheader("EDA")
                    lottie_he = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_j6bwzbtp.json")
                    st_lottie(
                        lottie_he,
                    )
                    st.subheader("data  plot ")
                    if st.checkbox("Show All DataFrame"):
                        st.dataframe(df.head())
                        st.write()

                    # Show Description
                    if st.checkbox("Show All Column Name"):
                        st.text("Columns:")
                        st.write(df.columns)

                    # Dimensions
                    data_dim = st.radio('What Dimension Do You Want to Show', ('Rows', 'Columns'))
                    if data_dim == 'Rows':
                        st.text("Showing Length of Rows")
                        st.write(len(df))
                    if data_dim == 'Columns':
                        st.text("Showing Length of Columns")
                        st.write(df.shape[1])

                    if st.checkbox("Show Summary of Dataset"):
                        st.write(df.describe())
                    if st.checkbox(" plot data"):
                        a = df['model'].value_counts()[:20]
                        st.bar_chart(a)
                        plt1 = df['brand'].value_counts()
                        st.bar_chart(plt1)

                        plt.title('Companies Histogram')
                        plt2 = df.groupby(['transmission'])['price'].mean()
                        st.bar_chart(plt2)
                    if st.checkbox("pie chart car model"):
                        input_co,pie_col=st.columns(2)

                        incol = input_co.text_input('how many top cars you want to see', 10)
                        incol = int(incol)
                        a = df['model'].value_counts()[:incol]
                        s = pd.Series(a)
                        mo = s.to_frame()
                        fig = px.pie(mo,values='model',names=mo.index)

                        pie_col.write(fig)
                    if st.checkbox("pie chart car brand"):
                        inpu_co, pie_cl = st.columns(2)

                        inco = inpu_co.text_input('how many top brand cars you want to see', 10)
                        inco = int(inco)
                        bran = df['brand'].value_counts()[:inco]
                        b = pd.Series(bran)
                        m = b.to_frame()
                        fi = px.pie(m, values='brand', names=m.index)
                        pie_cl.write(fi)





                    brand_list = ['vw', 'bmw', 'skoda', 'merc', 'hyundi', 'audi', 'toyota', 'ford']
                    g = st.selectbox("brand", brand_list)
                    if g == 'vw':
                        best_model('vw')

                    elif g == 'bmw':
                        best_model('bmw')
                    elif g == 'skoda':
                        best_model('skoda')
                    elif g == 'merc':
                        best_model('merc')

                    elif g == 'hyundi':
                        best_model('hyundi')
                    elif g == 'audi':
                        best_model('audi')
                    elif g == 'toyota':
                        best_model('toyota')
                    sca_list = ['year', 'mileage', 'engineSize']
                    sca = st.selectbox("line ", sca_list)
                    da = df[['year','price']]


                    if sca == 'year':
                        st.altair_chart(da)

















                elif activity == 'Prediction':

                    st.subheader("data vs plt ")
                    model_dic = {' 1 Series': 0, ' 2 Series': 1, ' 3 Series': 2, ' 4 Series': 3, ' 5 Series': 4,
                                 ' 6 Series': 5, ' 7 Series': 6, ' 8 Series': 7, ' A Class': 8, ' A1': 9, ' A2': 10,
                                 ' A3': 11, ' A4': 12, ' A5': 13, ' A6': 14, ' A7': 15, ' A8': 16, ' Accent': 17,
                                 ' Amarok': 18, ' Amica': 19, ' Arteon': 20, ' Auris': 21, ' Avensis': 22, ' Aygo': 23,
                                 ' B Class': 24, ' B-MAX': 25, ' Beetle': 26, ' C Class': 27, ' C-HR': 28, ' C-MAX': 29,
                                 ' CC': 30, ' CL Class': 31, ' CLA Class': 32, ' CLC Class': 33, ' CLK': 34,
                                 ' CLS Class': 35, ' Caddy': 36, ' Caddy Life': 37, ' Caddy Maxi': 38,
                                 ' Caddy Maxi Life': 39, ' California': 40, ' Camry': 41, ' Caravelle': 42,
                                 ' Citigo': 43, ' Corolla': 44, ' E Class': 45, ' EcoSport': 46, ' Edge': 47,
                                 ' Eos': 48, ' Escort': 49, ' Fabia': 50, ' Fiesta': 51, ' Focus': 52, ' Fox': 53,
                                 ' Fusion': 54, ' G Class': 55, ' GL Class': 56, ' GLA Class': 57, ' GLB Class': 58,
                                 ' GLC Class': 59, ' GLE Class': 60, ' GLS Class': 61, ' GT86': 62, ' Galaxy': 63,
                                 ' Getz': 64, ' Golf': 65, ' Golf SV': 66, ' Grand C-MAX': 67,
                                 ' Grand Tourneo Connect': 68, ' Hilux': 69, ' I10': 70, ' I20': 71, ' I30': 72,
                                 ' I40': 73, ' I800': 74, ' IQ': 75, ' IX20': 76, ' IX35': 77, ' Ioniq': 78,
                                 ' Jetta': 79, ' KA': 80, ' Ka+': 81, ' Kamiq': 82, ' Karoq': 83, ' Kodiaq': 84,
                                 ' Kona': 85, ' Kuga': 86, ' Land Cruiser': 87, ' M Class': 88, ' M2': 89, ' M3': 90,
                                 ' M4': 91, ' M5': 92, ' M6': 93, ' Mondeo': 94, ' Mustang': 95, ' Octavia': 96,
                                 ' PROACE VERSO': 97, ' Passat': 98, ' Polo': 99, ' Prius': 100, ' Puma': 101,
                                 ' Q2': 102, ' Q3': 103, ' Q5': 104, ' Q7': 105, ' Q8': 106, ' R Class': 107,
                                 ' R8': 108, ' RAV4': 109, ' RS3': 110, ' RS4': 111, ' RS5': 112, ' RS6': 113,
                                 ' RS7': 114, ' Ranger': 115, ' Rapid': 116, ' Roomster': 117, ' S Class': 118,
                                 ' S-MAX': 119, ' S3': 120, ' S4': 121, ' S5': 122, ' S8': 123, ' SL CLASS': 124,
                                 ' SLK': 125, ' SQ5': 126, ' SQ7': 127, ' Santa Fe': 128, ' Scala': 129,
                                 ' Scirocco': 130, ' Sharan': 131, ' Shuttle': 132, ' Streetka': 133, ' Superb': 134,
                                 ' Supra': 135, ' T-Cross': 136, ' T-Roc': 137, ' TT': 138, ' Terracan': 139,
                                 ' Tiguan': 140, ' Tiguan Allspace': 141, ' Touareg': 142, ' Touran': 143,
                                 ' Tourneo Connect': 144, ' Tourneo Custom': 145, ' Transit Tourneo': 146,
                                 ' Tucson': 147, ' Up': 148, ' Urban Cruiser': 149, ' V Class': 150, ' Veloster': 151,
                                 ' Verso': 152, ' Verso-S': 153, ' X-CLASS': 154, ' X1': 155, ' X2': 156, ' X3': 157,
                                 ' X4': 158, ' X5': 159, ' X6': 160, ' X7': 161, ' Yaris': 162, ' Yeti': 163,
                                 ' Yeti Outdoor': 164, ' Z3': 165, ' Z4': 166, ' i3': 167, ' i8': 168, '180': 169,
                                 '200': 170, '220': 171, '230': 172}
                    brand_dic = {'audi': 0, 'bmw': 1, 'ford': 2, 'hyundi': 3, 'merc': 4, 'skoda': 5, 'toyota': 6,
                                 'vw': 7}
                    ful_dic = {'Diesel': 0, 'Electric': 1, 'Hybrid': 2, 'Other': 3, 'Petrol': 4}

                    trans = {'Automatic': 0, 'Manual': 1, 'Other': 2, 'Semi-Auto': 3}
                    tran_list = ['Automatic', 'Manual', 'Other', 'Semi-Auto']
                    ful_list = ['Diesel', 'Electric', 'Hybrid', 'Other', 'Petrol']

                    brand_list = ['vw', 'bmw', 'skoda', 'merc', 'hyundi', 'audi', 'toyota', 'ford']
                    # Sidebar

                    # loading both models                         st.title('# Used Car Price Predition™  🚗')
                    st.markdown("<h2 style='text-align: center;'>🚗  Used Car Price Prediction™  🚗</h2>",
                                unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    mileage = col1.number_input(
                        label='Enter the how the car has driven, e.g: 200 mile (enter number only)',
                        help='how much the cas driven?')

                    # 3. year integer
                    year = col1.slider('Enter the year when the car was manufactured, e.g: 2005', 1980, 2020, 2005,
                                       help='The year when the car is manufactured.')

                    brand_inp = col1.selectbox(label='Enter the Brand of the car', options=brand_list,
                                               help='From which brand the car is belongs?')  # passing the brand list

                    brand = brand_dic[brand_inp]  # converting the brand name to numerical encoding form
                    ful_inp = col1.selectbox(label='Enter the Brand of the car', options=ful_list,
                                             help='From which brand the car is belongs?')  # passing the brand list
                    ful = ful_dic[ful_inp]  # converting the brand name to numerical encoding form
                    trans_inp = col2.selectbox(label='Enter the Brand of the car', options=tran_list,
                                               help='From which brand the car is belongs?')  # passing the brand list
                    trans = trans[trans_inp]
                    mpg = col2.slider('Enter the mpg, e.g: 60', 0.0, 480.0)
                    eng = col2.number_input(
                        label='Enter the how the eng type,',
                        help='how much the eng type ?')

                    # 8. model of the car for specific brand
                    if brand_inp == 'audi':
                        model_inp = col2.selectbox('Enter the Model for the Audi', options=find_model('audi'))
                        model = model_dic[model_inp]  # doing numerical encoding

                    elif brand_inp == 'vw':
                        model_inp = col2.selectbox('Enter the model for vw', options=find_model('vw'))
                        model = model_dic[model_inp]

                    elif brand_inp == 'hyundi':
                        model_inp = col2.selectbox('Enter the model Renault', options=find_model('hyundi'))
                        model = model_dic[model_inp]

                    elif brand_inp == 'toyota':
                        model_inp = col2.selectbox('Enter the model for Toyota', options=find_model('toyota'))
                        model = model_dic[model_inp]

                    elif brand_inp == 'bmw':
                        model_inp = col2.selectbox('Enter the model for BWM', options=find_model('bmw'))
                        model = model_dic[model_inp]

                    elif brand_inp == 'merc':
                        model_inp = col2.selectbox('Enter the model for Mercedes Benz',
                                                   options=find_model('merc'))
                        model = model_dic[model_inp]

                    elif brand_inp == 'skoda':
                        model_inp = col2.selectbox('Enter the model for Volkswagen', options=find_model('skoda'))
                        model = model_dic[model_inp]

                    predict = col1.button('Predict')  # creating a predict buutton



                    xtest = np.array([model, year, trans, mileage, ful, mpg, eng]).reshape(-1, 7)


                    if predict :
                     predicted = moodel.predict(xtest)
                     predicted=predicted*1.4
                     write = 'The predicted price of the car is $ ' + str(predicted) + ' 🚙'  # showing the price prediction.
                     st.success(write)

                     lottie_hel = load_lottieurl("https://assets8.lottiefiles.com/private_files/lf30_z1sghrbu.json")
                     st_lottie(
                         lottie_hel,
                     )



                    # Header of Specify Input Parameters
                    st.sidebar.header('Specify Input Parameters')

                    def user_input_features():

                        # Sidebar
                        # Header of Specify Input Parameters
                        st.sidebar.header('Specify Input Parameters')

                        def user_input_features():
                            # Sidebar
                            # Header of Specify Input Parameters
                            st.sidebar.header('Specify Input Parameters')



    elif choice == "Signup":
        new_username = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password", type='password')
        com_password = st.sidebar.text_input(" confirme Password", type='password')
        if new_password == com_password:
            st.success("password confirmed ")
            lottie_hell = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_gktj7rsr.json")
            st_lottie(
                lottie_hell,
            )

        else:
            st.success("password not the same ")
        if st.button("submit"):
            create_usertabel()
            new_password_hashed = generate_hashes(new_password)
            add_userdata(new_password, new_password_hashed)
            st.success("You have successfully created a new account")
            st.info("Login to Get Started")
            pass


if __name__ == '__main__':
    main()

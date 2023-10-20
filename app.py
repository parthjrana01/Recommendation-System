import pickle
import streamlit as st
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np

df = pickle.load(open('products.pkl','rb'))

X = df[['ratings','no_of_ratings', 'discount_price', 'actual_price']]
y = df.cluster

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X, y)

st.header("Product Recommendation system")
# python -m streamlit run app.py 

def recommend_product(product):
    product_detail = df[df['name'] == product]
    
    c = knn.predict([[product_detail.iloc[0][5],product_detail.iloc[0][6],product_detail.iloc[0][7],product_detail.iloc[0][8]]])[0]
    return df[df['cluster'] == c].head()

selected_products = st.selectbox(
    "Type or select a product",
    df["name"]
)



if st.button('Show Recommendation'):
    print(selected_products)
    recommendation_products = recommend_product(selected_products)
    col1,col2,col3,col4,col5=st.columns(5)

    with col1:
        st.text(recommendation_products.iloc[1][0])
        st.image(recommendation_products.iloc[1][3])

    with col2:
        st.text(recommendation_products.iloc[2][0])
        st.image(recommendation_products.iloc[2][3])

    with col3:
        st.text(recommendation_products.iloc[3][0])
        st.image(recommendation_products.iloc[3][3])

    with col4:
        st.text(recommendation_products.iloc[4][0])
        st.image(recommendation_products.iloc[4][3])

    with col5:
        st.text(recommendation_products.iloc[0][0])
        st.image(recommendation_products.iloc[0][3])
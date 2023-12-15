import streamlit as st
import pandas as pd
import pickle
import numpy as np
import folium
from streamlit_folium import st_folium

header = st.container()
dataset = st.container()
features = st.container()
model = st.container()

with header:
    st.title("Bienvenue dans mon projet de Machine Learning")
    st.text("Ce projet permet d'analyser les routes à hauts risques d'accident")

with dataset:
    def load_model():
        with open('saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)

        return data


data = load_model()

le_cluster= data["model"]
le_departement= data["le_departement"]
le_region = data["le_region"]
le_saison = data["le_saison"]

with features:
    st.title("Prédiction d'accidents de routes en France Métropolitaine")

    st.write("""### Nous avons besoin de quelques renseignements pour prédire les accidents""")

    region = (
            "Île-de-France",
            "Auvergne-Rhône-Alpes"
            "Provence-Alpes-Côte d'Azur",
            "Nouvelle-Aquitaine",
            "Occitanie",
            "Grand Est",
            "Hauts-de-France",  
            "Normandie",                    
            "Bretagne",                    
            "Pays de la Loire",             
            "Bourgogne-Franche-Comté" ,   
            "Centre-Val de Loire",        
            "Corse",
        )
    
    saison = (
            "Hiver",
            "Printemps",
            "Ete",
            "Automne",
        )    

    region = st.selectbox("Région", (region))
    saison = st.selectbox("Saison", (saison))

    ok = st.button("Routes accidentelles")
    if ok:

        map = folium.Map(location=[46.232193, 2.209667], zoom_start=9, tiles="openstreetmap")
       
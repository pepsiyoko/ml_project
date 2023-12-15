import streamlit as st
import pickle
import numpy as np
import pandas as pd 

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

model_loaded = data["model"]
le_departement= data["le_departement"]
le_region = data["le_region"]
le_saison = data["le_saison"]

def show_predict_page():
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
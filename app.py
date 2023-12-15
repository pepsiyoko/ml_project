import streamlit as st
import pickle
import numpy as np


def load_model():
    with open("/home/pepsiyoko/code/pepsiyoko/project/mlapp/caracteristiques-2022.csv", "r") as csv_file:
        headers = "".join([line.strip() for line in csv_file.readlines()[:2]]).split(";")
        data = pickle.load(csv_file)
    return data

data = load_model()

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

    region = st.selectbox("Région", region)
    saison = st.selectbox("Saison", saison)
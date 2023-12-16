import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import folium
import streamlit.components.v1 as components

header = st.container()
dataset = st.container()
features = st.container()
model = st.container()

with header:
    st.title("Bienvenue dans mon projet de Machine Learning")
    st.text("Ce projet permet d'analyser les lieux à hauts risques d'accident")

df = pd.read_csv("Accident.csv")

# Encoder les colonnes 'Région' et 'Saison'
label_encoder_region = LabelEncoder()
label_encoder_saison = LabelEncoder()

df['Région_encoded'] = label_encoder_region.fit_transform(df['Région'])
df['Saison_encoded'] = label_encoder_saison.fit_transform(df['Saison'])

# Sélectionner les features et les cibles
features = ['Région_encoded', 'Saison_encoded']
targets = ['Latitude', 'Longitude']

X = df[features]
y = df[targets]

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de régression linéaire pour la latitude
model_latitude = LinearRegression()
model_latitude.fit(X_train, y_train['Latitude'])

# Entraîner un modèle de régression linéaire pour la longitude
model_longitude = LinearRegression()
model_longitude.fit(X_train, y_train['Longitude'])

# Définir les options des menus déroulants pour la région et la saison
regions_options = df['Région'].unique()
saisons_options = df['Saison'].unique()

st.title("Prédiction d'accidents de routes en France Métropolitaine")
st.write("""### Nous avons besoin de quelques renseignements pour prédire les accidents""")

# Ajouter des widgets pour les menus déroulants
selected_region = st.selectbox('Choisissez une région:', regions_options)
selected_saison = st.selectbox('Choisissez une saison:', saisons_options)

# Filtrer les données en fonction des choix de l'utilisateur
df_selected = df[(df['Région'] == selected_region) & (df['Saison'] == selected_saison)]

# Faire des prédictions sur les données filtrées
X_selected = df_selected[features]
predictions_latitude = model_latitude.predict(X_selected)
predictions_longitude = model_longitude.predict(X_selected)

# Ajouter les prédictions au DataFrame sélectionné
df_selected['Prédiction_Latitude'] = predictions_latitude
df_selected['Prédiction_Longitude'] = predictions_longitude

# Afficher les prédictions sur une carte avec folium
def afficher_predictions_sur_carte(df, coordonnees_col, prediction_lat_col, prediction_lon_col):
    # Créer une carte centrée sur la première coordonnée de la liste
    carte = folium.Map(location=df[coordonnees_col].iloc[0], zoom_start=10)

    # Ajouter des marqueurs pour chaque coordonnée avec une couleur différente en fonction des prédictions
    for _, row in df.iterrows():
        folium.CircleMarker(location=row[coordonnees_col],
                        radius=2,
                        popup=f"Latitude: {row[prediction_lat_col]:.6f}, Longitude: {row[prediction_lon_col]:.6f}",
                      icon=folium.Icon(color='blue')).add_to(carte)

    # Afficher la carte dans le navigateur
    st.components.v1.html(carte._repr_html_(), height=500)

# Utiliser la fonction pour afficher les prédictions sur une carte
afficher_predictions_sur_carte(df_selected, ['Latitude', 'Longitude'], 'Prédiction_Latitude', 'Prédiction_Longitude')

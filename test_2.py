import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import folium

# Charger le DataFrame
df = pd.read_csv("votre_fichier.csv")  # Assurez-vous de remplacer "votre_fichier.csv" par le nom de votre fichier CSV

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

# Faire des prédictions sur l'ensemble de test
predictions_latitude = model_latitude.predict(X_test)
predictions_longitude = model_longitude.predict(X_test)

# Ajouter les prédictions au DataFrame de test
df_test = df.loc[X_test.index].copy()
df_test['Prédiction_Latitude'] = predictions_latitude
df_test['Prédiction_Longitude'] = predictions_longitude

# Afficher le DataFrame avec les prédictions
st.write(df_test[['Région', 'Saison', 'Latitude', 'Longitude', 'Prédiction_Latitude', 'Prédiction_Longitude']])

# Afficher les prédictions sur une carte avec folium
def afficher_predictions_sur_carte(df, coordonnees_col, prediction_lat_col, prediction_lon_col):
    # Créer une carte centrée sur la première coordonnée de la liste
    carte = folium.Map(location=df[coordonnees_col].iloc[0], zoom_start=10)

    # Ajouter des marqueurs pour chaque coordonnée avec une couleur différente en fonction des prédictions
    for _, row in df.iterrows():
        folium.Marker(location=row[coordonnees_col], 
                      popup=f"Latitude: {row[prediction_lat_col]:.6f}, Longitude: {row[prediction_lon_col]:.6f}",
                      icon=folium.Icon(color='blue')).add_to(carte)

    # Afficher la carte dans le navigateur
    carte.save('carte_predictions.html')
    st.markdown("## Carte des prédictions")
    st.components.v1.html(open('carte_predictions.html').read(), height=500)

# Utiliser la fonction pour afficher les prédictions sur une carte
afficher_predictions_sur_carte(df_test, ['Latitude', 'Longitude'], 'Prédiction_Latitude', 'Prédiction_Longitude')

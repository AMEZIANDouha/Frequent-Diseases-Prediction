import pickle
import csv
import streamlit as st
import pandas as pd
import joblib
import pickle
from streamlit_option_menu import option_menu
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# Read Raw Dataset
df = pd.read_csv(r"C:\Users\amezi\OneDrive\Bureau\Dis_Pred\data\preprocessed_data.csv")

# Sélectionner les valeurs uniques de la colonne 'Disease'
unique_diseases = df['Disease'].unique()

# Sélectionner les valeurs uniques de toutes les colonnes de symptômes
unique_symptoms = df.iloc[:, :-1].stack().unique()

# Créer un label encoder pour les maladies
label_encoder_disease = LabelEncoder()
# Appliquer le label encoding aux maladies
encoded_labels_disease = label_encoder_disease.fit_transform(unique_diseases)
encoding_dict_disease = dict(zip(unique_diseases, encoded_labels_disease))

# Créer un label encoder pour les symptômes
label_encoder_symptom = LabelEncoder()
# Appliquer le label encoding aux symptômes
encoded_labels_symptom = label_encoder_symptom.fit_transform(unique_symptoms)
encoding_dict_symptom = dict(zip(unique_symptoms, encoded_labels_symptom))

# Appliquer le label encoding à la colonne 'Disease' dans le dataframe
df['Disease'] = df['Disease'].map(encoding_dict_disease)

X = df.iloc[:, :-1]
y = df['Disease']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True ,stratify=y)

# Initialiser le SVM modèle
model =  KNeighborsClassifier(n_neighbors=3)

# Entraîner le modèle
model.fit(X, y)

# Define custom CSS styles
custom_styles = """
<style>
body {
    background-color: "red";  /* Set the background color */
}
=
.sidebar .sidebar-content {
    background-color: #4CAF50;  /* Set the sidebar background color */
}
div.stButton > button {
    background-color: #4CAF50;  /* Set the button background color */
    color: white;  /* Set the button text color */
    font-weight: bold;  /* Make the button text bold */
}
</style>
"""

# Apply custom CSS styles
st.markdown(custom_styles, unsafe_allow_html=True)

# Sauvegarder le modèle
joblib.dump(model, 'knn_model.pkl')

# Sidebar for navigation
with st.sidebar:
    selected = st.selectbox('Navigation', ['Home', 'Disease Prediction'])

    st.markdown("<div style='position: fixed; bottom: 10px; left: 30px; font-size: 12px;font-weight: bold'>Disease Prediction PUI Made By AMEZIAN DOUHA</div>", unsafe_allow_html=True)


# Home Page
if selected == 'Home':
    # Définir le style CSS pour l'arrière-plan de la page Home avec une taille de cadre personnalisée
    background_style = """
        <style>
            body {
                background-color: #8B4513; /* Marron */
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0; /* Supprimer la marge par défaut du corps */
            }
            .custom-container {
                padding: 20px;
                border-radius: 10px;
                background-color: #4682B4; /* Bleu acier */
                color: white;
                text-align: center;
                max-width: 100%; /* Largeur maximale du cadre (90% de la largeur de l'écran) */
                max-height: 100%; /* Hauteur maximale du cadre (80% de la hauteur de l'écran) */
                overflow-y: auto; /* Ajouter une barre de défilement si le contenu est trop grand */
            }
        </style>
    """
    
    # Appliquer le style CSS
    st.markdown(background_style, unsafe_allow_html=True)
    
    # Contenu de la page Home
    st.markdown("""
       <div class="custom-container">
           <h1>Welcome to Disease Prediction App</h1>
           <p>This app is a product of a comprehensive machine learning project that involved a comparative study of various algorithms created by Douha Amezian! The vision of this project is to provide a user-friendly platform for predicting diseases based on symptoms. By leveraging machine learning, we aim to assist users in understanding potential health issues and taking proactive measures. If you have any questions, feedback, or need assistance, feel free to reach out to me at <a href="mailto:Ameziandouha5@gmail.com">Ameziandouha5@gmail.com</a>.</p>
           <p>Your well-being is our priority!</p>
       </div>
   """, unsafe_allow_html=True)


# Disease Prediction Page
if selected == 'Disease Prediction':
    # Page title
    st.title('Disease Prediction using ML')
    

    # Getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        symptom1 = st.selectbox('Symptom 1', sorted(df.columns[:-1]))

        symptom3 = st.selectbox('Symptom 3', sorted(df.columns[:-1]))

        symptom5 = st.selectbox('Symptom 5', sorted(df.columns[:-1]))

    with col2:
        symptom2 = st.selectbox('Symptom 2', sorted(df.columns[:-1]))

        symptom4 = st.selectbox('Symptom 4', sorted(df.columns[:-1]))

        symptom6 = st.selectbox('Symptom 6', sorted(df.columns[:-1]))

    # Code for Prediction
    disease_diagnosis = ''
    button_color = "#4CAF50"
    # Creating a button for Prediction
    if st.button('Disease Test Result'):
        # Create a new DataFrame for prediction with all zeros
        input_data = pd.DataFrame(np.zeros((1, len(df.columns)-1), dtype=int), columns=df.columns[:-1])

        # Set the selected symptoms to 1
        input_data.at[0, symptom1] = 1
        input_data.at[0, symptom2] = 1
        input_data.at[0, symptom3] = 1
        input_data.at[0, symptom4] = 1
        input_data.at[0, symptom5] = 1
        input_data.at[0, symptom6] = 1

        # Make the prediction
        disease_prediction = model.predict(input_data)

        # Convert the encoded prediction back to the original disease label
        predicted_disease = label_encoder_disease.inverse_transform(disease_prediction)[0]

        # Display the result
        st.success(f'The predicted disease is: {predicted_disease}')
        
        
        # Centered and moved down butto___
    st.markdown(
        """<style>
        div.stButton > button {
            
            
            font-weight: bold;
            display: block;
            margin: 0 auto;
            margin-top: 40px;
        }
        </style>""",
        unsafe_allow_html=True
    )
#st.markdown("<div style='position: fixed; bottom: 10px; left: 10px; font-size: 12px;'>AMEZIAN DOUHA</div>", unsafe_allow_html=True)
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS  # Importar la extensión CORS
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from chatbot import generar_respuesta, libros_df, personajes_df
import spacy
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
import ast
from textblob import TextBlob
import nltk
import deepl
from TextoImagenApiPrueba2 import recomendadoraimagen, generate_images_from_text
from gensim.models import LdaModel
import os
from flask_caching import Cache
from Codigo import process_image_and_recommend
import tempfile
import torch
import csv

# Inicializacion
# Crear un cliente DeepL con tu clave de API
auth_key = "76f60294-03cb-41d1-9bda-69a352564258:fx"  # Reemplaza con tu clave de API
deepl_client = deepl.DeepLClient(auth_key)
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__, static_folder='../public')

CORS(app)

# Configuración de la caché
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Funciones necesarias

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val  # Devuelve el valor original si no se puede evaluar

def unir_lista(lista):
    return ', '.join(lista)

def limpieza_df(df, stopwords=False, stemming=False, lemmatization=False):
    # Se limpia la columna de descripcion y se guarda en una nueva columna llamada cleaned_desc
    df['cleaned_desc'] = df['description'].apply(_removeNonAscii)
    df['cleaned_desc'] = df.cleaned_desc.apply(func=make_lower_case)
    df['cleaned_desc'] = df.cleaned_desc.apply(func=remove_punctuation)
    df['cleaned_desc'] = df.cleaned_desc.apply(func=remove_html)
    if stopwords:
        df['cleaned_desc_stopwords'] = df.cleaned_desc.apply(func=remove_stop_words)
        if stemming:
            df['cleaned_desc_stemming'] = df.cleaned_desc_stopwords.apply(func=apply_stemming)
        if lemmatization:
            df['cleaned_desc_lemmatization'] = df.cleaned_desc_stopwords.apply(func=apply_lemmatization)
        return df
    else:
        if stemming:
            df['cleaned_desc_stemming'] = df.cleaned_desc.apply(func=apply_stemming)
        if lemmatization:
            df['cleaned_desc_lemmatization'] = df.cleaned_desc.apply(func=apply_lemmatization)
        return df

# Funcion para eliminar los NonAscii characters
def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)

# Funcion para convertir a minusculas
def make_lower_case(text):
    return text.lower()

# Funcion para eliminar las stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

# Función para eliminar la puntuación
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

# Función para eliminar html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Función para realizar stemming
def apply_stemming(text):
    stemmer = PorterStemmer()
    text = text.split()
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text

# Función para lemmatization
def apply_lemmatization(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    text = " ".join(lemmas)
    return text

def unir_lista(lista):
    return ', '.join(lista)

def traduccion_español(texto):
    return deepl_client.translate_text(texto, target_lang="ES").text

# CORRER SI YA SE TENIA LOS DATOS
df = pd.read_csv('base_datos.csv')
df = df.dropna(subset=['description'])  # Se elimina los libros que no tengan descripcion
df = df.fillna('')  # Se cambia los valores NaN por string vacios
df['genres'] = df['genres'].apply(safe_literal_eval)
df['title_lower'] = df['title'].apply(make_lower_case)

def recommend_description_lematizacion(description, genre='No', top_n=5):
    description = deepl_client.translate_text(description, target_lang="EN-US").text
    # Matching the genre with the dataset and reset the index
    if genre != 'No':
        data = df[df['genres'].apply(lambda x: genre in x)]
    else:
        data = df.copy()

    data.reset_index(level=0, inplace=True)
    # Convert the index into series
    indices = pd.Series(data.index, index=data['title'])

    # Convert the book descriptions into vectors using TfidfVectorizer
    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(data['cleaned_desc_lemmatization'])
    description_cleaned = _removeNonAscii(description)
    description_cleaned = make_lower_case(description_cleaned)
    description_cleaned = remove_punctuation(description_cleaned)
    description_cleaned = apply_lemmatization(description_cleaned)

    # Transform the input description into the same vector space
    desc_vector = tf.transform([description_cleaned])

    # Calculate the similarity measures based on Cosine Similarity
    sg = cosine_similarity(desc_vector, tfidf_matrix)
    # Obtener los índices de los top_n libros más similares
    top_indices = sg.argsort()[0][-top_n:][::-1]

    # Obtener los títulos de los libros más similares
    recommended_books = data[['title', 'description', 'author', 'genres', 'pages', 'publishDate','cover_image_uri']].iloc[top_indices].to_dict('records')

    # Convertir la lista de géneros a una cadena de texto separada por comas y limpiar las páginas
    for book in recommended_books:
        book['title_es'] = traduccion_español(book['title'])
        book['description_es'] = traduccion_español(book['description'])
        if isinstance(book['genres'], list):
            book['genres'] = ', '.join(book['genres'])
        else:
            book['genres'] = str(book['genres'])
        book['pages'] = str(book['pages']).replace("[", "").replace("]", "").replace("'", "")

    return recommended_books

def recommend_title(title, genre='No', top_n=5):
    title = deepl_client.translate_text(title, target_lang="EN-US").text
    title = make_lower_case(title)
    print(title)

    if genre != 'No':
        data = df[df['genres'].apply(lambda x: genre in x)]
    else:
        data = df.copy()

    data.reset_index(level=0, inplace=True)
    indices = pd.Series(data.index, index=data['title'])

    if title in df['title_lower'].values:
        description_cleaned = df[df['title_lower'] == title]['cleaned_desc_lemmatization'].values[0]
    else:
        print(f"El título '{title}' no se encontró en el dataset. Recomendando con el un título similar")
        tf_title = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1, stop_words='english')
        tfidf_matrix_title = tf_title.fit_transform(data['title_lower'])
        title_vector = tf_title.transform([title])
        sg_title = cosine_similarity(title_vector, tfidf_matrix_title)
        top_indices_title = sg_title.argsort()[0][-1:][::-1]
        title = data['title_lower'].iloc[top_indices_title].values[0]
        description_cleaned = df[df['title_lower'] == title]['cleaned_desc_lemmatization'].values[0]

    print(title)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(data['cleaned_desc_lemmatization'])
    desc_vector = tf.transform([description_cleaned])

    sg = cosine_similarity(desc_vector, tfidf_matrix)
    top_indices = sg.argsort()[0][-top_n:][::-1]

    recommended_books = data[['title', 'description', 'author', 'genres', 'pages', 'publishDate', 'cover_image_uri']].iloc[top_indices].to_dict('records')

    for book in recommended_books:
        book['title_es'] = traduccion_español(book['title'])
        book['description_es'] = traduccion_español(book['description'])
        if isinstance(book['genres'], list):
            book['genres'] = ', '.join(book['genres'])
        else:
            book['genres'] = str(book['genres'])
        book['pages'] = str(book['pages']).replace("[", "").replace("]", "").replace("'", "")

    return recommended_books

def get_book_descriptions(titles):
    descriptions = []
    for title in titles:
        if len(df[df['title'] == title]) == 0:
            print(f"Descripción no encontrada para el título: {title}")
            continue
        description = df[df['title'] == title]['description'].values[0]
        descriptions.append(description)
    return descriptions


# Ruta principal
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Ruta para recibir la descripción y devolver recomendaciones
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    description = data['description']
    genre = data.get('genre', 'No')  # Opcional: filtrar por género

    # Obtener recomendaciones
    recommended_books = recommend_description_lematizacion(description, genre)
    
    return jsonify({'recommended_books': recommended_books})

@app.route('/recommend_by_title', methods=['POST'])
def recommend_by_title():
    data = request.json
    title = data['title']
    genre = data.get('genre', 'No')  # Optional: filter by genre

    # Get recommendations
    recommended_books = recommend_title(title, genre)

    return jsonify({'recommended_books': recommended_books})

# Generador de Imagenes
@app.route('/generated_images/<filename>')
def serve_generated_image(filename):
    return send_from_directory(os.path.join(app.root_path, '..', 'generated_images'), filename)

@app.route('/generate_images_for_book', methods=['POST'])
def generate_images_for_book():
    data = request.json
    book_title = data['book_title']
    
    # Generar URLs de imágenes a partir del título del libro
    image_urls = recomendadoraimagen([book_title])
    
    return jsonify({'image_urls': image_urls})

@app.route('/generate_images', methods=['POST'])
def generate_images():
    data = request.json
    description = data['description']  # Descripción de la imagen proporcionada por el usuario

    # Llamar a la función que genera imágenes (recomendadoraimagen)
    image_urls = generate_images_from_text(description)  # Pasar la descripción como una lista

    return jsonify({'image_urls': image_urls})

#Chatbot

# Contexto global de la conversación
contexto_conversacion = []
personaje_actual = None

@app.route('/chat', methods=['POST'])
def chat():
    global personaje_actual, contexto_conversacion

    # Verificar los datos recibidos en Flask
    data = request.get_json()
    print("📩 Datos recibidos en Flask:", data)

    # Obtener y limpiar el input del usuario
    user_input = data.get('message', '').strip()
    book_id = data.get('bookId')
    book_title = data.get('bookTitle')  # Obtener el título del libro
    book_title = deepl_client.translate_text(book_title, target_lang="ES").text
    book_title = make_lower_case(book_title)
    print(book_title)
    
        
    print(f"📝 Entrada del usuario en Flask: '{user_input}'")  # Depuración

    if not user_input and not book_title:
        return jsonify({"response": "No recibí ningún mensaje. ¿Puedes intentarlo de nuevo?"})

    if "adios" in user_input.lower():
        return jsonify({"response": "¡Hasta luego! Espero haberte ayudado."})
    
    if user_input != '' and personaje_actual is None:
       personaje_actual= str(user_input)
    
    if user_input == '':
        personaje_actual = None
        contexto_conversacion = []
        
    if personaje_actual is None:
        # Obtener el título del libro basado en el bookId si no se proporciona book_title
        libro = book_title if book_title else libros_df.loc[libros_df['id'] == book_id, 'Titulo'].values[0]
        print(libro)
        
        # Obtener los personajes disponibles para el libro
        personajes_disponibles = personajes_df[personajes_df['Libro'].apply(make_lower_case) == libro]['Nombre del Personaje'].tolist()
        
        if personajes_disponibles:
            personajes_str = ', '.join(personajes_disponibles)
            return jsonify({"response": f"¿Con qué personaje desearías hablar? Los personajes disponibles son: {personajes_str}"})
        else:
            return jsonify({"response": "No se encontraron personajes para este libro."})
        
    else:
        # Si ya se ha seleccionado un personaje, verificar si el usuario está seleccionando un personaje
        personajes_disponibles = personajes_df[personajes_df['Libro'] == book_title]['Nombre del Personaje'].tolist()
        if user_input in personajes_disponibles:
            personaje_actual = str(user_input)
            return jsonify({"response": f"Has seleccionado a {personaje_actual}. ¿En qué puedo ayudarte?"})
        else:
            # Continuar la conversación con el personaje seleccionado
            try:
                # Imprimir el mensaje antes de pasarlo al chatbot
                print(f"🤖 Generando respuesta para: '{user_input}'")
                print(f"💬 generada: {personaje_actual}")
                print("📜 Contexto actual:", contexto_conversacion)
                # Generar respuesta del chatbot
                respuesta = generar_respuesta(personaje_actual, user_input, contexto_conversacion)
                print(f"💬 Respuesta generada: {respuesta}")  # Verificar salida

                # Actualizar el contexto
                contexto_conversacion.append({"role": "user", "content": user_input})
                contexto_conversacion.append({"role": "assistant", "content": respuesta})

                # Limitar el contexto a los últimos 10 intercambios
                if len(contexto_conversacion) > 10:
                    contexto_conversacion = contexto_conversacion[-10:]

                # Mostrar el contexto actualizado
                print("📜 Contexto actual:", contexto_conversacion)
                
                respuesta = deepl_client.translate_text(respuesta, target_lang="ES").text
                return jsonify({"response": respuesta})
            
            except Exception as e:
                print("❌ Error en la generación de respuesta:", e)
                return jsonify({"response": "Lo siento, no puedo procesar esa pregunta en este momento."})

#recomendador apartir de imagenes
# Ruta absoluta al archivo del modelo
MODEL_PATH = os.path.abspath("my_model.pt")

# Verificar si el archivo del modelo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {MODEL_PATH}")

# Cargar el modelo
model = torch.load(MODEL_PATH)

# Ruta para manejar la carga de imágenes y la recomendación de libros
@app.route('/recommend_from_image', methods=['POST'])
def recommend_from_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Crear un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image_path = temp_image.name
            file.save(image_path)

        # Procesar la imagen y obtener recomendaciones
        indices_libros_recomendados = process_image_and_recommend(image_path)
        recommended_books = df[['title', 'description', 'author', 'genres', 'pages', 'publishDate', 'cover_image_uri']].iloc[indices_libros_recomendados].to_dict('records')

        for book in recommended_books:
            book['title_es'] = traduccion_español(book['title'])
            book['description_es'] = traduccion_español(book['description'])
            if isinstance(book['genres'], list):
                book['genres'] = ', '.join(book['genres'])
            else:
                book['genres'] = str(book['genres'])
            book['pages'] = str(book['pages']).replace("[", "").replace("]", "").replace("'", "")

        
        return jsonify({"recommended_books": recommended_books})
    
    except Exception as e:
        # Registrar el error en la consola del servidor
        print(f"Error en recommend_from_image: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    finally:
        # Eliminar la imagen temporal
        if os.path.exists(image_path):
            os.remove(image_path)
            
# Ruta al archivo CSV
CSV_FILE = 'ratings.csv'

# Crear el archivo CSV si no existe
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User Input', 'Book Description', 'Rating'])

@app.route('/rate', methods=['POST'])
def rate():
    data = request.json
    user_input = data.get('sentence1')
    book_description = data.get('sentence2')
    rating = data.get('score')

    # Guardar la calificación en el archivo CSV
    with open(CSV_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_input, book_description, rating])

    return jsonify({'message': 'Calificación guardada correctamente'}), 200

if __name__ == '__main__':
    app.run(debug=True)
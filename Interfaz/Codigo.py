import cv2
from ultralytics import YOLO
import requests
import json
import deepl
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import tempfile
import os

# Inicialización de DeepL
auth_key = "76f60294-03cb-41d1-9bda-69a352564258:fx"  # Reemplaza con tu clave de API
deepl_client = deepl.DeepLClient(auth_key)
model_recomendacion = SentenceTransformer('version2')
doc_embeddings = np.load("doc_embeddings.npy")
libros_extraidos = []
libros_a_recomendar = []
indices_a_recomendar = []
libros_recomendados = []
indices_libros_recomendados = []

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val  # Devuelve el valor original si no se puede evaluar

def make_lower_case(text):
    return text.lower()

# Cargar el dataset de libros
df = pd.read_csv('base_datos.csv')
df = df.dropna(subset=['description'])  # Eliminar libros sin descripción
df = df.fillna('')  # Reemplazar NaN con strings vacíos
df['genres'] = df['genres'].apply(ast.literal_eval)
df['title_lower'] = df['title'].apply(lambda x: x.lower())

# Cargar el modelo YOLO
MODEL_PATH = "my_model.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Función para OCR
def ocr_space_file(filename, overlay=False, api_key='K83117384188957', language='eng'):
    payload = {'isOverlayRequired': overlay, 'apikey': api_key, 'language': language}
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image', files={filename: f}, data=payload)
    return r.content.decode()

def images_to_text(image_path):
    test_file = ocr_space_file(image_path, language='spa')
    # Cargar el JSON
    data = json.loads(test_file)

    # Extraer el texto
    parsed_text = data["ParsedResults"][0]["ParsedText"]
    print("Texto crudo:", parsed_text)

    # Limpiar el texto
    cleaned_text = parsed_text.replace("\r\n", " ")
    print("Texto limpio:", cleaned_text)
    return cleaned_text

# Función para buscar un libro parecido
def buscar_libro_parecido(title,lang='es'):
    if title == '':
        print("No se puede recomendar un libro sin un título")
        return
    if lang == 'es':
        title = deepl_client.translate_text(title, target_lang="EN-US").text
        print(f"Traducción: {title}")
    title = make_lower_case(title)

    data = df.copy()
    data.reset_index(level=0, inplace=True)
    indices = pd.Series(data.index, index=data['title'])

    tf_title = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), min_df=1)
    tfidf_matrix_title = tf_title.fit_transform(data['title_author'])
    title_vector = tf_title.transform([title])
    sg_title = cosine_similarity(title_vector, tfidf_matrix_title)
    top_indices_title = sg_title.argsort()[0][-1:][::-1]
    title = data['title_lower'].iloc[top_indices_title].values[0]
    print(f"Libro encontrado: {title}")
    description_cleaned = df[df['title_lower'] == title]['cleaned_desc_lemmatization'].values[0]

    return top_indices_title

# Función para procesar la imagen y obtener recomendaciones
def process_image_and_recommend(image_path: str):
    indices_libros_recomendados = [] #reinicio de variables
    libros_a_recomendar = []
    indices_a_recomendar = []
    libros_recomendados = []  # Declarar dentro de la función
    try:
        # Procesar la imagen con YOLO
        results = model.predict(source=image_path, conf=0.5)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen desde {image_path}")

        libros_extraidos = []
        for i, box in enumerate(results[0].boxes):
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            
            # Clase y confianza de la detección
            class_id = int(box.cls)
            confidence = box.conf.item()
            print(f"Objeto {i + 1}:")
            print(f"  Clase: {class_id}, Confianza: {confidence:.2f}")
            print(f"  Coordenadas: x_min={x_min:.0f}, y_min={y_min:.0f}, x_max={x_max:.0f}, y_max={y_max:.0f}")

            h, w = image.shape[:2]
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

            book_image = image[y_min:y_max, x_min:x_max]
            output_path = f"book{i + 1}.jpg"
            cv2.imwrite(output_path, book_image)

            extracted_text = images_to_text(output_path)
            libros_extraidos.append(extracted_text)

        # Obtener recomendaciones basadas en los textos extraídos
        for text in libros_extraidos:
            indice = buscar_libro_parecido(text)
            if indice is not None:  # Verificar que se haya encontrado una recomendación válida
                libro = df['title'].iloc[indice].values[0]
                libros_a_recomendar.append(libro)
                indices_a_recomendar.append(indice)
        for indice in indices_a_recomendar:
            print(f"Indices: {indice}")
                        
        documents = df['description'].tolist()
        
        for indice in indices_a_recomendar:
            query_embedding = model_recomendacion.encode(df['description'].iloc[indice].values[0])
            # Find the most similar document to the query
            results = util.semantic_search(query_embedding, doc_embeddings, top_k=2)
            most_similar_doc = documents[results[0][0]['corpus_id']]
            libro_recomendado = df.iloc[results[0][1]['corpus_id']]['title']
            libros_recomendados.append(libro_recomendado)
            indices_libros_recomendados.append(results[0][1]['corpus_id'])
        
        for libro in libros_recomendados:
            print(f"Libro recomendado: {libro}")
        return indices_libros_recomendados
    except Exception as e:
        print(f"Error en process_image_and_recommend: {str(e)}")
        return []
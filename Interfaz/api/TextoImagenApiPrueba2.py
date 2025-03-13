import sys
import io
import os
from gradio_client import Client
from PIL import Image
import re
from flask import url_for
import httpx
import base64
import tempfile
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import sent_tokenize
import deepl
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Crear cliente
client = Client("black-forest-labs/FLUX.1-schnell", hf_token="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")

def textoimagen(prompt):
    global client  # Declarar la variable client como global
    if client is None:
        print("Gradio client no fue inicializado, pero lo estamos inicializando.")
        try:
            client = Client("black-forest-labs/FLUX.1-schnell", hf_token="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")
        except ValueError as e:
            print(f"Error initializing Gradio client: {e}")
            return None

    try:
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=512,
            height=512,
            num_inference_steps=2,
            api_name="/infer"
        )
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

    # Verificar si el resultado es una ruta local de imagen
    image_path, _ = result
    
    # Leer la imagen y codificarla en base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:image/png;base64,{encoded_string}"  # Devolver la imagen en formato base64

#---------------------------------------------------PARA EL MODELO DE LDA---------------------------------------------------
nltk.download('punkt')
# Descargar y cargar el modelo de spaCy para inglés
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
# Establecer las palabras vacías
stop_words = STOP_WORDS

# Definir la función de preprocesamiento
def preprocess(text):
    # Procesar el texto con spaCy
    doc = nlp(text.lower())
    # Filtrar tokens
    filtered_tokens = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV'] and token.text not in stop_words and token.is_alpha]
    return filtered_tokens

#-------Esto es para leer el DatasetPreprocesado que se usará para entrenar el modelo LDA----------------
Directoriodatapreprocesado = "DatasetPreprocesado.csv"
datapreprocesado = pd.read_csv(Directoriodatapreprocesado)
datapreprocesado = datapreprocesado[["title", "description","processed"]]
import ast

datapreprocesado['processed'] = datapreprocesado['processed'].apply(ast.literal_eval)
# Crear un diccionario y un corpus
processed_docs = datapreprocesado['processed'].tolist()
dictionary = Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Verificar si el modelo LDA preentrenado existe
lda_model_path = 'lda_model.gensim'
if os.path.exists(lda_model_path):
    # Cargar el modelo LDA desde el archivo
    lda_model = LdaModel.load(lda_model_path)
    print("Modelo LDA cargado desde el archivo.")
else:
    # Entrenar el modelo LDA
    lda_model = LdaModel(corpus, num_topics=50, id2word=dictionary, passes=3)
    # Guardar el modelo LDA entrenado
    lda_model.save(lda_model_path)
    print("Modelo LDA entrenado y guardado en el archivo.")
    
# Obtenemos las distribuciones de palabras para cada tópico
topics = lda_model.get_topics()
# Calcula la similitud del coseno entre tópicos

similarity_matrix = cosine_similarity(topics)
# Aplica K-means clustering

num_clusters = 10  # Número de categorías de libros

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(similarity_matrix)
# Asigna cada tópico a un cluster
clusters = kmeans.labels_

# Muestra los tópicos agrupados
for i in range(num_clusters):
    print(f"Categoría {i + 1}:")
    topics_in_cluster = np.where(clusters == i)[0]
    for topic_idx in topics_in_cluster:
        print(f"Tópico {topic_idx}: {lda_model.print_topic(topic_idx, topn=5)}")
    print("\n")

def Generartopico(test_text):
    # Preprocesamiento del texto
    test_text_tokens = preprocess(test_text)
    test_text_bow = dictionary.doc2bow(test_text_tokens)

    # Obtén la distribución de tópicos para el texto
    test_text_topics = lda_model.get_document_topics(test_text_bow)

    # Verifica los tópicos obtenidos
    print("Tópicos obtenidos:", test_text_topics)

    # Encuentra el tópico con mayor probabilidad
    if test_text_topics:
        top_topic = max(test_text_topics, key=lambda item: item[1])

        # Obtén una palabra del tópico más relevante
        top_word = lda_model.show_topic(top_topic[0], topn=1)[0][0]
        print(f"Etiqueta: {top_word}")
    else:
        print("No se encontraron tópicos relevantes para el texto de prueba.")
    return top_word

#---------------------------------------------------PARA DIVIDIR EL TEXTO LARGO EN MAS PEQUEÑOS ---------------------------------------------------
def split_text_large_input(text,                 # Texto largo a separar
                           min_chunk_length=150,  # Mínimo de caracteres por chunk
                           split_method="smart",  # Opciones: "sentence", "paragraph", "fixed"
                           overlap_sentences=1,   # Solapamiento entre chunks
                           random_start=False):   # Comenzar desde una posición aleatoria

    # Preprocesamiento: eliminar espacios extras y saltos de línea múltiples
    if isinstance(text, pd.Series):
        text = text.apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
        text = text.tolist()  # Convertir Series a lista para facilitar el manejo
    else:
        text = re.sub(r'\s+', ' ', text.strip())
    
    if isinstance(text, list):
        text = ' '.join(text)
    
    # Comenzar desde una posición aleatoria si random_start es True
    if random_start:
        text_length = len(text)
        start_pos = random.randint(0, max(0, text_length - min_chunk_length))
        text = text[start_pos:]

    # Método 1: Dividir por oraciones naturales
    if split_method == "sentence":
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) <= min_chunk_length or not current_chunk:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                if current_length < min_chunk_length:
                    additional_sentence = sentences.pop(0)
                    current_chunk.append(additional_sentence)
                    current_length += len(additional_sentence)
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)

        if current_chunk:
            if current_length < min_chunk_length and len(sentences) > 0:
                additional_sentence = sentences.pop(0)
                current_chunk.append(additional_sentence)
                current_length += len(additional_sentence)
            chunks.append(" ".join(current_chunk))

    # Método 2: Dividir por párrafos (doble salto de línea)
    elif split_method == "paragraph":
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []

        for para in paragraphs:
            if len(para) > min_chunk_length:
                sub_chunks = split_text_large_input(para, min_chunk_length, "sentence")
                chunks.extend(sub_chunks)
            else:
                chunks.append(para)

    # Método 3: División fija por longitud
    elif split_method == "fixed":
        chunks = [text[i:i+min_chunk_length] for i in range(0, len(text), min_chunk_length)]

    # Método "smart": Combina párrafos y oraciones
    else:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []

        for para in paragraphs:
            if len(para) <= min_chunk_length:
                chunks.append(para)
            else:
                sentences = sent_tokenize(para)
                current_chunk = []
                current_length = 0

                for sentence in sentences:
                    if current_length + len(sentence) <= min_chunk_length or not current_chunk:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                    else:
                        if current_length < min_chunk_length:
                            additional_sentence = sentences.pop(0)
                            current_chunk.append(additional_sentence)
                            current_length += len(additional_sentence)
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                if current_chunk:
                    if current_length < min_chunk_length and len(sentences) > 0:
                        additional_sentence = sentences.pop(0)
                        current_chunk.append(additional_sentence)
                        current_length += len(additional_sentence)
                    chunks.append(" ".join(current_chunk))

    # Agregar solapamiento entre chunks
    if overlap_sentences > 0 and split_method == "sentence":
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                overlap = " ".join(sent_tokenize(chunks[i-1])[-overlap_sentences:])
                overlapped_chunks.append(overlap + " " + chunks[i])
            else:
                overlapped_chunks.append(chunks[i])
        chunks = overlapped_chunks

    return chunks

#---------------------------------------------------PARA PASAR EL TEXTO ---------------------------------------------------
# Crear un cliente DeepL con tu clave de API
auth_key = "76f60294-03cb-41d1-9bda-69a352564258:fx"  # Reemplaza con tu clave de API
deepl_client = deepl.DeepLClient(auth_key)

def pasartexto(chunks, recomendador):
    image_paths = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\n=== Chunk {i} ===")
        translation = deepl_client.translate_text(chunk, target_lang="EN-US")
        print(translation.text)
        print(f"Longitud: {len(translation.text)} caracteres")
        etiqueta = Generartopico(translation.text)
        translation.text = translation.text + "," + etiqueta
        prompt = translation.text
        print(prompt)
        image_path = textoimagen(prompt)
        image_paths.append(image_path)
        if recomendador:
            break  # Detener la iteración después de la primera si recomendador es True

    return image_paths

DirectorioEncontrarDescipcion = "books_cleaned_descripcion_char_places.csv"
EncontrarDescipcion = pd.read_csv(DirectorioEncontrarDescipcion)
EncontrarDescipcion = EncontrarDescipcion[["title", "description"]]

#---------------------------------------------------PARA EL RECOMENDADOR DE IMAGENES---------------------------------------------------
def clean_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)

def recomendadoraimagen(recommended_books):
    image_urls = []
    generated_images_dir = os.path.join(os.path.dirname(__file__), '..', 'generated_images')
    os.makedirs(generated_images_dir, exist_ok=True)

    for book in recommended_books:
        clean_book_title = clean_filename(book)
        for idx in range(3):  # Generar tres imágenes por libro
            image_filename = f"{clean_book_title}_{idx + 1}.png"
            image_path = os.path.join(generated_images_dir, image_filename)

            # Verificar si la imagen ya existe
            if os.path.exists(image_path):
                # Agregar la URL de la imagen existente a la lista de URLs
                image_urls.append(url_for('serve_generated_image', filename=image_filename, _external=True))
            else:
                # Obtener la descripción del libro
                descripcion = EncontrarDescipcion[EncontrarDescipcion["title"] == book]["description"]
                if len(descripcion) == 0:
                    chunks = [book]  # Usar el título del libro si no hay descripción
                else:
                    chunks = split_text_large_input(descripcion.iloc[0], min_chunk_length=150, split_method="smart", overlap_sentences=1, random_start=True)
                
                # Generar la imagen
                base64_image = get_base64_image_for_book(chunks[idx % len(chunks)], idx)
                if base64_image is None:
                    continue  # Si no se pudo generar la imagen, continuar con el siguiente libro

                # Decodificar la cadena base64 y guardar la imagen
                image_data = base64.b64decode(base64_image.split(',')[1])
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_data)

                # Agregar la URL de la imagen procesada a la lista de URLs
                image_urls.append(url_for('serve_generated_image', filename=image_filename, _external=True))

    return image_urls

def get_base64_image_for_book(text_chunk, idx):
    # Utiliza la función Generartopico para obtener la etiqueta del modelo LDA
    etiqueta = Generartopico(text_chunk)
    
    # Crear el prompt con la etiqueta
    prompt = f"{text_chunk}, {etiqueta}"
    
    # Utiliza la función textoimagen para generar la imagen basada en el título del libro y la etiqueta
    base64_image = textoimagen(prompt)
    return base64_image

def generate_images_from_text(text):
    image_urls = []
    generated_images_dir = os.path.join(os.path.dirname(__file__), '..', 'generated_images')
    os.makedirs(generated_images_dir, exist_ok=True)

    # Limpiar el nombre del archivo para evitar caracteres no válidos
    clean_text = clean_filename(text[:50])  # Usar los primeros 50 caracteres para el nombre del archivo

    # Dividir el texto en partes usando split_text_large_input
    text_chunks = split_text_large_input(text, min_chunk_length=150, split_method="smart", overlap_sentences=1, random_start=True)

    # Asegurarse de que haya al menos 3 partes (si el texto es corto, se repite el último chunk)
    while len(text_chunks) < 3:
        text_chunks.append(text_chunks[-1])  # Repetir el último chunk si no hay suficientes

    # Generar 3 imágenes basadas en los chunks de texto
    for idx in range(4):  # Generar tres imágenes
        image_filename = f"{clean_text}_{idx + 1}.png"
        image_path = os.path.join(generated_images_dir, image_filename)

        # Obtener el chunk de texto correspondiente
        chunk = text_chunks[idx % len(text_chunks)]  # Usar el módulo para evitar errores si hay menos de 3 chunks

        # Generar la imagen usando el chunk de texto
        base64_image = textoimagen(chunk)
        if base64_image is None:
            continue  # Si no se pudo generar la imagen, continuar con la siguiente

        # Decodificar la cadena base64 y guardar la imagen
        image_data = base64.b64decode(base64_image.split(',')[1])
        with open(image_path, 'wb') as image_file:
            image_file.write(image_data)

        # Agregar la URL de la imagen procesada a la lista de URLs
        image_urls.append(url_for('serve_generated_image', filename=image_filename, _external=True))

    return image_urls

# Calcular la perplexity
perplexity = lda_model.log_perplexity(corpus)
print("La perplexity es una métrica comúnmente utilizada para evaluar modelos de topicos. Mide cuán bien el modelo predice una muestra. Una menor perplexity indica un mejor modelo. Sin embargo, esta métrica no siempre correlaciona con la calidad interpretable de los tópicos.")
print(f'Perplexity: {perplexity}')

#prueba = ["harry potter y la piedra filosofal","harry potter y la camara secreta","harry potter y el prisionero de azkaban","harry potter y el caliz de fuego","harry potter y la orden del fenix"]
#print(recomendadoraimagen(prueba))

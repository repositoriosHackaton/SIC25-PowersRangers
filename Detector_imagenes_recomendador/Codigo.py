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

auth_key = "76f60294-03cb-41d1-9bda-69a352564258:fx"  # Reemplaza con tu clave de API
deepl_client = deepl.DeepLClient(auth_key)

model_recomendacion = SentenceTransformer('version2')
doc_embeddings = np.load("doc_embeddings.npy")

libros_extraidos = []
libros_a_recomendar = []
indices_a_recomendar = []
libros_recomendados = []

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val  # Devuelve el valor original si no se puede evaluar


def ocr_space_file(filename, overlay=False, api_key='K86486204688957', language='eng'):

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()

def images_to_text(image_path):
    test_file = ocr_space_file(image_path, language='spa')
    # Cargar el JSON
    data = json.loads(test_file)

    # Extraer el texto
    parsed_text = data["ParsedResults"][0]["ParsedText"]

    # Limpiar el texto
    cleaned_text = parsed_text.replace("\r\n", " ")
    return cleaned_text

def recommend_title(title,lang='es'):
    if title == '':
        print("No se puede recomendar un libro sin un título")
        return
    if lang == 'es':
        title = deepl_client.translate_text(title, target_lang="EN-US").text
        title = make_lower_case(title)
    data = df.copy()

    data.reset_index(level=0, inplace=True)

    tf_title = TfidfVectorizer(analyzer='char', ngram_range=(2, 2), min_df=1)
    tfidf_matrix_title = tf_title.fit_transform(data['title_author'])
    title_vector = tf_title.transform([title])
    sg_title = cosine_similarity(title_vector, tfidf_matrix_title)
    top_indices_title = sg_title.argsort()[0][-1:][::-1]
    title = data['title_lower'].iloc[top_indices_title].values[0]

    return top_indices_title


# Funcion para convertir a minusculas
def make_lower_case(text):
    return text.lower()

# CORRER SI YA SE TENIA LOS DATOS
df = pd.read_csv('base_datos.csv')
df = df.dropna(subset=['description'])  # Se elimina los libros que no tengan descripcion
df = df.fillna('')  # Se cambia los valores NaN por string vacios
df['genres'] = df['genres'].apply(safe_literal_eval)
df['title_lower'] = df['title'].apply(make_lower_case)

# Carga tu modelo YOLO (puede ser uno preentrenado o personalizado)
model = YOLO("my_model.pt")  # Sustituye "best.pt" por tu modelo YOLO

# Ejecuta la predicción sobre la imagen de entrada
results = model.predict(source="test1.jpg", conf=0.5)

# Ruta de la imagen original
image_path = "test1.jpg"

# Cargar la imagen
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen desde {image_path}")

# Procesar cada detección y recortar los libros
for i, box in enumerate(results[0].boxes):
    # Extraer las coordenadas del bounding box
    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

    # Clase y confianza de la detección
    class_id = int(box.cls)
    confidence = box.conf.item()

    print(f"Objeto {i + 1}:")
    print(f"  Clase: {class_id}, Confianza: {confidence:.2f}")

    # Asegurar que las coordenadas estén dentro de los límites de la imagen
    h, w = image.shape[:2]
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

    # Recortar la región de interés (ROI)
    book_image = image[y_min:y_max, x_min:x_max]

    # Guardar la imagen recortada con el formato book1.jpg, book2.jpg, ...
    output_path = f"book{i + 1}.jpg"
    cv2.imwrite(output_path, book_image)


        # Pasar la imagen guardada a la función images_to_text
    extracted_text = images_to_text(output_path)

    # Agregar el texto extraído a la lista
    libros_extraidos.append(extracted_text)

# Imprimir los textos extraídos
print("\nTextos extraídos de las imágenes:")
for idx, text in enumerate(libros_extraidos, start=1):
    print(f"Libro {idx}: {text}")

# Recomendar un libro similar al texto extraído
for text in libros_extraidos:
    indice = recommend_title(text)
    libro = df['title'].iloc[indice].values[0]
    libros_a_recomendar.append(libro)
    indices_a_recomendar.append(indice)

print("\nTextos a recomendar:")

for libro in libros_a_recomendar:
    print(f"Libro: {libro}")


documents = df['description'].tolist()


# Encode query and documents
for indice in indices_a_recomendar:
    query_embedding = model_recomendacion.encode(df['description'].iloc[indice].values[0])
    # Find the most similar document to the query
    results = util.semantic_search(query_embedding, doc_embeddings, top_k=2)
    most_similar_doc = documents[results[0][0]['corpus_id']]
    libro_recomendado = df.iloc[results[0][1]['corpus_id']]['title']
    libros_recomendados.append(libro_recomendado)

print("\nLibros recomendados:")

for libro in libros_recomendados:
    print(f"Libro recomendado: {libro}")


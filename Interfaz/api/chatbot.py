import pandas as pd
from groq import Groq
import os

# Configuración de la API de OpenAI
api_key = "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"
client = Groq(api_key=api_key)
model = "mixtral-8x7b-32768"

# Verificación de archivos Excel
if not os.path.isfile("Biblioteca_libros.xlsx") or not os.path.isfile("Biblioteca_personajes.xlsx"):
    print("Error: Archivos Excel no encontrados.")
    exit(1)

# Carga de datos
try:
    libros_df = pd.read_excel("Biblioteca_libros.xlsx")
    personajes_df = pd.read_excel("Biblioteca_personajes.xlsx")
    
    if libros_df.empty or personajes_df.empty:
        print("Error: Los archivos Excel están vacíos.")
        exit(1)
except Exception as e:
    print(f"Error al cargar los archivos Excel: {str(e)}")
    exit(1)

def extraer_personaje_y_libro(texto):
    """Extrae nombre de personaje y libro del texto del usuario"""
    texto = texto.lower().strip()
    
    if " del libro " in texto:
        partes = texto.split(" del libro ")
        return partes[0].strip(), partes[1].strip()
    elif " en " in texto:
        partes = texto.split(" en ")
        return partes[0].strip(), partes[1].strip()
    
    return texto.strip(), None

def buscar_personaje_por_libro(nombre, libro=None):
    """Busca personaje por nombre y libro"""
    try:
        filtro_nombre = personajes_df['Nombre del Personaje'].str.contains(nombre, case=False, na=False)
        resultados = personajes_df[filtro_nombre]
        if libro:
            filtro_libro = resultados['Libro'].str.contains(libro, case=False, na=False)
            resultados = resultados[filtro_libro]
        return resultados.iloc[0] if not resultados.empty else None
    except Exception as e:
        print(f"Error al buscar personaje: {str(e)}")
        return None

def buscar_libro(titulo):
    """Busca libros por título"""
    try:
        resultados = libros_df[libros_df['Titulo'].str.contains(titulo, case=False, na=False)]
        return resultados.iloc[0] if not resultados.empty else None
    except Exception as e:
        print(f"Error al buscar libro: {str(e)}")
        return None

def generar_respuesta(personaje, pregunta, contexto_conversacion=None):
    """Genera respuesta como el personaje manteniendo el contexto histórico."""
    try:
        # Asegúrate de que 'personaje' es un diccionario y no una cadena
        if isinstance(personaje, str):
            personaje = buscar_personaje_por_libro(personaje)

        prompt = f"""
        Eres {personaje['Nombre del Personaje']}, un personaje de la literatura.
        Tu personalidad es: {personaje['Personalidad']}.
        Responde como si narraras tus propias experiencias y pensamientos.
        Si no tienes información específica sobre el tema, inventa una respuesta coherente y acorde a tu carácter.
        No expliques tu proceso de pensamiento ni muestres detalles internos o etiquetas como <think>.
        Responde directamente como el personaje, sin desviarte.
        Limita tu respuesta a un máximo de 3 oraciones, y asegúrate de que sea completa, inventando detalles si es necesario.
        No devuelvas únicamente signos de puntuación.
        Nunca te salgas del personaje cuando te pregunten algo que el personaje desconoce, tampoco aclares que eres un personaje de ficcion.
        """
        
        # Construir la lista de mensajes en orden cronológico:
        mensajes = [{"role": "system", "content": prompt}]
        if contexto_conversacion:
            mensajes.extend(contexto_conversacion)
        mensajes.append({"role": "user", "content": pregunta})
        
        response = client.chat.completions.create(
            model=model,
            messages=mensajes
        )
        
        respuesta_filtrada = response.choices[0].message.content.strip()
        print(f"Respuesta sin filtrar: {respuesta_filtrada}")
        # Remover posibles etiquetas internas
        if "<think>" in respuesta_filtrada:
            respuesta_filtrada = respuesta_filtrada.split("<think>")[0].strip()
        
        # Procesar la respuesta: separar oraciones y limitar a 3 oraciones
        partes = [p.strip() for p in respuesta_filtrada.split(".") if p.strip()]
        respuesta_final = ". ".join(partes[:3])
        if respuesta_final:
            respuesta_final += "."
        
        if len(respuesta_final.strip()) <= 1 or respuesta_final.strip() == ".":
            respuesta_final = "Lo siento, no tengo una respuesta completa en este momento. ¿Podrías reformular la pregunta?"
        
        return respuesta_final
    except Exception as e:
        print(f"Error al generar respuesta: {str(e)}")
        return "Lo siento, no puedo procesar esa pregunta en este momento."

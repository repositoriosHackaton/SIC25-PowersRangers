DreamPages

Sistema de recomendación de libros impulsado por NLP, que ofrece recomendaciones personalizadas basadas en descripciones detalladas de los usuarios.

## Tabla de contenidos

1. [Nombre](#Nombre)
2. [Descripción](#descripción)
3. [Arquitectura](#Arquitectura)
4. [Proceso](#Proceso)
5. [Funcionalidades](#Funcionalidades)
6. [Estado del proyecto](#EstadoDelProyecto)
7. [Agradecimientos](#Agradecimientos)

* Nombre
  
DreamPages

* Descripcíon
  
Sistema de recomendación de libros impulsado por NLP, que ofrece recomendaciones personalizadas basadas en descripciones detalladas de los usuarios. -> Alguna imagen o gif que muestre el proyecto


* Arquitectura del proyecto + imagen
  
El sistema consta de cuatro componentes principales: el frontend, el backend, el chatbot y la API de imágenes.

-Frontend (App.tsx): Es la interfaz de usuario donde se ingresan datos (descripción o título del libro), se muestran recomendaciones, detalles del libro, se interactúa en el chat y se visualizan imágenes adicionales.

-Backend (app.py): Procesa los datos ingresados por el usuario, genera recomendaciones de libros, gestiona el chat y genera imágenes adicionales. Actúa como intermediario entre el frontend, el chatbot y la API de imágenes.

-Chatbot (chatbot.py): Genera respuestas basadas en el contexto de la conversación del chat.

-API de Imágenes (TextoImagenApiPrueba2.py): Genera imágenes basadas en el título del libro proporcionado.

![flujo](https://github.com/user-attachments/assets/b8593ac7-8102-4996-a7d2-7e3dc99fab53)




* Proceso de desarrollo:

-Fuente del dataset
Diferentes Datasets provenientes de Kaggel

-Limpieza de datos (img que lo valide)
![code](https://github.com/user-attachments/assets/85264278-51e7-4184-8ecf-ea9046e834e1)


-Manejo excepciones/control errores
En el codigo hay "Exception" que se imprimen en el terminal indicando cual es error que ocurre

-Estadísticos (Valores, gráficos, …)

![grafica1](https://github.com/user-attachments/assets/e48c1af4-44a2-4284-a98f-78b71885667f)

![Grafica](https://github.com/user-attachments/assets/532a2ee9-ed2e-46ae-a7b9-316961da7894)

* Funcionalidades extra:

-Integración del proyecto en una pág web
- Tecnología/Herramientas usadas …
  Flask, Html, Css, javascript, react
- Arquitectura (img)
  ![code](https://github.com/user-attachments/assets/360627d0-0f0f-42f7-ad07-f6d9c249d8f8)


import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Mostrar versión de Python
st.caption(f" Versión de Python en uso: {platform.python_version()}")

# Cargar el modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título principal
st.title("📸 Reconocimiento de Imágenes con Teachable Machine")

# Imagen de referencia o logo
image = Image.open('image_2025-10-23_002544528.png')
st.image(image, width=350, caption="Ejemplo de detección o modelo base")

# Información lateral
with st.sidebar:
    st.header("ℹ️ Información")
    st.write("""
    Esta aplicación utiliza un modelo entrenado en **Teachable Machine**  
    para reconocer imágenes en tiempo real.
    
    Puedes tomar una foto desde la cámara y ver qué clase identifica el modelo.
    """)

# Entrada de la cámara
img_file_buffer = st.camera_input("📷 Toma una foto para analizar")

if img_file_buffer is not None:
    # Leer imagen desde la cámara
    img = Image.open(img_file_buffer)

    # Redimensionar imagen al tamaño requerido por el modelo
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir imagen a array numpy
    img_array = np.array(img)

    # Normalizar los valores de la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Ejecutar la predicción
    prediction = model.predict(data)

    # Mostrar resultados
    st.subheader("🔍 Resultado del modelo:")

    if prediction[0][0] > 0.5:
        st.success(f"➡️ Clase: **Izquierda** | Probabilidad: {round(prediction[0][0], 2)}")
    if prediction[0][1] > 0.5:
        st.success(f"⬆️ Clase: **Arriba** | Probabilidad: {round(prediction[0][1], 2)}")
    # if prediction[0][2] > 0.5:
    #     st.success(f"➡️ Clase: **Derecha** | Probabilidad: {round(prediction[0][2], 2)}")

# Pie de página
st.caption("Desarrollado usando Streamlit, Keras y Teachable Machine.")

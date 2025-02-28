import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array




#PATHS
modelo_path = 'modelo_figuras.h5'
ruta_imagen_nueva = 'orgimg/testc.jpg'


#CARGA EL MODELO
modelo = tf.keras.models.load_model(modelo_path)
#MAPEO ETIQUETAS
mapeo_etiquetas = {0: "Círculo", 1: "Cuadrado"}



#NORMALIZACION Y PREDICCION
def predecir_imagen(ruta_imagen):
    # Cargar y procesar la imagen
    imagen = load_img(ruta_imagen, target_size=(190, 190))  # Cambiar tamaño al usado en entrenamiento
    imagen_array = img_to_array(imagen)  # Convertir a array numpy
    imagen_array = imagen_array / 255.0  # Normalizar la imagen (0-1)
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir dimensión batch

    # Realizar la predicción
    prediccion = modelo.predict(imagen_array)
    clase_predicha = np.argmax(prediccion)  # Obtener la clase con mayor probabilidad
    probabilidad = np.max(prediccion)  # Obtener la probabilidad de la clase predicha

    # Imprimir el resultado
    print(f"Predicción: {mapeo_etiquetas[clase_predicha]} (Probabilidad: {probabilidad*100:.2f}%)")
    return mapeo_etiquetas[clase_predicha], probabilidad




#LLAMA A FUNCION
resultado, prob = predecir_imagen(ruta_imagen_nueva)

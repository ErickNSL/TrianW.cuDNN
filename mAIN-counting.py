import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json



#PATHS
modelo = tf.keras.models.load_model('modelo_figuras.h5')



# Función para preprocesar y clasificar la imagen
def clasificar_imagen(ruta_imagen):
    # Cargar la imagen
    img = image.load_img(ruta_imagen, target_size=(190, 190))  # Asegúrate de usar el tamaño correcto
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir el eje para el lote
    
    # Normalizar la imagen (si es necesario)
    img_array = img_array / 255.0  # Si las imágenes se normalizaron entre 0 y 1 durante el entrenamiento

    # Hacer la predicción
    predicciones = modelo.predict(img_array)

    # Obtener la etiqueta con la probabilidad más alta
    clase_predicha = np.argmax(predicciones, axis=1)[0]

    # Devolver la clase predicha (0: círculo, 1: cuadrado, 2: otra)
    return clase_predicha

# Función para contar círculos y cuadrados en una imagen
def contar_figuras(ruta_imagen):
    # Inicializar contadores
    contador_circulos = 0
    contador_cuadrados = 0

    # Aquí puedes dividir la imagen en partes más pequeñas si es necesario,
    # para obtener varias predicciones (si tienes muchas figuras en la imagen)
    # Por ahora se usa la imagen completa para hacer una única predicción.

    clase = clasificar_imagen(ruta_imagen)
    if clase == 0:
        contador_circulos += 1
    elif clase == 1:
        contador_cuadrados += 1

    return contador_circulos, contador_cuadrados

# Ruta de la imagen a probar
ruta_imagen = 'orgimg/test.jpg'

# Contar las figuras en la imagen
circulos, cuadrados = contar_figuras(ruta_imagen)

# Mostrar los resultados
print(f"Circulos: {circulos}")
print(f"Cuadrados: {cuadrados}")

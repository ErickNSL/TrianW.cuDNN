import matplotlib.pyplot as plt
import numpy as np




############################################
### MUESTRA LAS IMAGENES INDIVIDUALMENTE ###
############################################



# Path to the .npy file
npy_file_path = "normalizedimg/cuadrado2.npy"
npy_file_path2 = "normalizedimg/t1o.npy"
# Load the .npy file
image_data = np.load(npy_file_path)

# Display the image
plt.imshow(image_data)  # Automatically scales the normalized [0, 1] pixel values
plt.axis('off')  # Hide axis labels for clarity
plt.title("Normalized Image")
plt.show()




#############################################################
### PARTE DE CODIGO PARA VERIFICAR LA INSERCION DE NUEVAS ###
### IMAGENES EN EL ENTRENAMIENTO DEL MODELO AUGMENTATION  ###
#############################################################



import matplotlib.pyplot as plt

# Obtener un lote de imágenes aumentadas
#x_batch, y_batch = next(train_generator)  # Obtener el siguiente lote del generador

# Mostrar algunas imágenes aumentadas
#num_images_to_show = 10  # Número de imágenes a visualizar
#plt.figure(figsize=(15, 5))

#for i in range(num_images_to_show):
    #plt.subplot(1, num_images_to_show, i + 1)
    #plt.imshow(x_batch[i].astype('float32'))  # Mostrar imagen (asegurarse de que esté en el rango correcto)
    #plt.title(f"Label: {y_batch[i]}")
    #plt.axis('off')

#plt.tight_layout()
#plt.show()

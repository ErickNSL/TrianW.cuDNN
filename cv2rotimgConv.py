import numpy as np
import cv2
import json
import os
import re




################################################################
### NORMALIZA LAS IMAGENES MODIFICADAS CON ROTACIONES        ###
### Y AGREGA LAS IMAGENES YA RODATAS AL DATASE  INCLUYE TAGS ###
################################################################




print("STARTING ### ...")


### PATHS
input_Dir = "orgimg/"                           #IMAGENES ORIGINALES
output_Dir = "normalizedimg/"                   #IMAGENES NORMALIZADAS
labels_File = "labels.json"                     #TAGS RELACIONADAS
os.makedirs(output_Dir, exist_ok=True)
target_Size = (190,190)



### ROTACION DE IMAGEN
def rotate_img(image,angle):

    center = (image.shape[1] // 2, image.shape[0] // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center,angle,1.0)

    rotated_image = cv2.warpAffine(image,rotation_matrix,(image.shape[1],image.shape[0]))

    return rotated_image



### NORMALIZACION DE IMAGEN
def normalization_augmentation(image_Path,save_Dir,label,angles=(70,140,250,280)):

    image = cv2.imread(image_Path)
    if image is None:
        print("Error loadding image...")
        return []
    
    resized_Img = cv2.resize(image, target_Size)
    norm_Img = resized_Img / 255.0


    base_name = os.path.splitext(os.path.basename(image_Path))[0]
    original_save_path = os.path.join(save_Dir, f"{base_name}.npy")
    np.save(original_save_path, norm_Img)
    augmented_files = [(original_save_path, label)]

    for angle in angles:
        rotated_img = rotate_img(resized_Img, angle)
        rotated_img = rotated_img / 255.0  # Normalize the rotated image
        rotated_save_path = os.path.join(save_Dir, f"{base_name}_rot{angle}.npy")
        np.save(rotated_save_path, rotated_img)
        augmented_files.append((rotated_save_path, label))

    return augmented_files



### MAPPING LABELS PARA JSON
label_Mapping = {
    "circulo": 0,
    "cuadrado": 1
}



### LOOP PARA CADA UNA DE LAS IMAGENES LLAMANDO FUNCION DE NORMALIZACION Y GUARDANDO EN LA VARABLE 
all_Labels = []
for file_name in os.listdir(input_Dir):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        base_name = os.path.splitext(file_name)[0]
        figure_name = re.sub(r'\d+', '', base_name).lower()# remuebe los digitos en los nombres para mayor capacidad de imagenes
        label = label_Mapping.get(figure_name.lower(), -1)
        if label == -1:
            print(f"{figure_name}")
            print(f"Unknown label for file: {file_name}")
            continue

        input_path = os.path.join(input_Dir, file_name)
        augmented_files = normalization_augmentation(input_path, output_Dir, label)
        all_Labels.extend(augmented_files)



### CONFIGURACION Y ESCRITURA DEL JSON
with open(labels_File, "w") as f:
    json.dump(all_Labels, f, indent=4)

print("Done!")




    


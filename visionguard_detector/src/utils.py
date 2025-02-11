import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def preprocess_image(image):
    # Função para pré-processar a imagem antes da detecção ou treinamento.
    return cv2.resize(image, (416, 416)) / 255.0   # Normalização como exemplo.

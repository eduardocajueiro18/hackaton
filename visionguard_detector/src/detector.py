import os

# Configuração do TensorFlow para evitar avisos desnecessários
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Apenas mostra erros críticos

import cv2
import numpy as np
import tensorflow as tf
from alert import send_alert  # Importa a função de alerta

# Definições do modelo
# Caminho dos videos
VIDEO_DIR = "../videos/"
# Caminho do modelo salvo
MODEL_PATH = "../models/final_model.keras"
# Dimensão esperada das imagens
IMG_SIZE = (416, 416)
# Limite de confiança para detectar objetos
CONFIDENCE_THRESHOLD = 0.5

# Carregar o modelo treinado
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Erro: O modelo não foi encontrado em '{MODEL_PATH}'.")

model = tf.keras.models.load_model(MODEL_PATH)
# Exibir estrutura do modelo para depuração
model.summary()

# Classes do modelo
CLASSES = ["faca", "objeto_cortante"]

def preprocess_frame(frame):
    # Redimensiona a imagem para o tamanho esperado pelo modelo
    frame = cv2.resize(frame, (416, 416))

    # Converte para um array NumPy e normaliza para valores entre 0 e 1
    frame = np.array(frame, dtype=np.float32) / 255.0

    # Expande a dimensão para corresponder ao formato (1, 416, 416, 3)
    frame = np.expand_dims(frame, axis=0)

    return frame

def detect_objects(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Erro: O vídeo não foi encontrado em '{video_path}'.")

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pré-processa o frame antes de passar para o modelo
        processed_frame = preprocess_frame(frame)

        print(f"Shape da imagem antes da predição: {processed_frame.shape}")  # Para depuração

        # Faz a predição
        predictions = model.predict(processed_frame)

        # Obtém a classe com maior probabilidade
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            label = CLASSES[class_id]
            color = (0, 255, 0) if label == "normal" else (0, 0, 255)

            if label in CLASSES:
                print("🚨 Objeto cortante detectado!")
                send_alert(frame)  # Chama a função de alerta

            cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"  # Substitua pelo caminho do vídeo a ser analisado
    detect_objects(os.path.join(VIDEO_DIR, video_path))

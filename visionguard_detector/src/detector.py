import os

# Configuração do TensorFlow para evitar avisos desnecessários
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Apenas mostra erros críticos

import cv2
import numpy as np
import tensorflow as tf
from alert import send_alert  # Importa a função de alerta

# Definições do modelo
VIDEO_DIR = "../videos/"
MODEL_PATH = "../models/final_model.keras"  # Caminho do modelo salvo
IMG_SIZE = (416, 416)  # Dimensão esperada das imagens
CONFIDENCE_THRESHOLD = 0.5  # Limite de confiança para detectar objetos

# Carregar o modelo treinado
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Erro: O modelo não foi encontrado em '{MODEL_PATH}'.")

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()  # Exibir estrutura do modelo para depuração

# Classes do modelo (ajuste conforme o seu dataset)
CLASSES = ["normal", "objeto_cortante"]  # Substitua pelos nomes corretos


def preprocess_frame(frame):
    """Pré-processa um frame antes de enviá-lo para o modelo."""
    frame_resized = cv2.resize(frame, IMG_SIZE)  # Redimensiona para o tamanho correto
    frame_normalized = frame_resized.astype("float32") / 255.0  # Normaliza para [0,1]
    frame_expanded = np.expand_dims(frame_normalized, axis=0)  # Adiciona dimensão do batch
    return frame_expanded


def detect_objects(video_path):
    # Carregar o modelo treinado
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Erro: O vídeo não foi encontrado em '{video_path}'.")

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pré-processa o frame antes de passar para o modelo
        processed_frame = preprocess_frame(frame)

        # **Correção: Garantir que processed_frame tenha a forma correta**
        processed_frame = processed_frame.reshape(1, -1)  # Converte (1, 416, 416, 3) para (1, 50176)

        print(f"Shape da imagem antes da predição: {processed_frame.shape}")  # Para depuração

        # Faz a predição
        predictions = model.predict(processed_frame)

        # Obtém a classe com maior probabilidade
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            label = CLASSES[class_id]
            color = (0, 255, 0) if label == "normal" else (0, 0, 255)

            if label == "objeto_cortante":
                print("🚨 Objeto cortante detectado!")
                send_alert(frame)  # Chama a função de alerta

            cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Detecção", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"  # Substitua pelo caminho do vídeo a ser analisado
    detect_objects(os.path.join(VIDEO_DIR, video_path))

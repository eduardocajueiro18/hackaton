import os

# Configuração do TensorFlow para suprimir avisos desnecessários
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Apenas mostra erros críticos

import cv2
import numpy as np
import tensorflow as tf
from alert import send_alert  # Função de alerta para objetos perigosos

VIDEOS_DIR = "../videos/"
# Caminhos do modelo treinado
MODEL_PATH = "../models/final_modelo.keras"
CLASS_NAMES = ["faca", "objeto_cortante"]

# Carregar modelo treinado
model = tf.keras.models.load_model(MODEL_PATH)

# Parâmetros de pré-processamento
IMG_SIZE = (416, 416)

def preprocess_frame(frame):
    """Pré-processa o frame para ser compatível com o modelo."""
    image = cv2.resize(frame, IMG_SIZE)  # Redimensiona para o tamanho esperado
    image = np.array(image, dtype=np.float32) / 255.0  # Normaliza os pixels
    image = np.expand_dims(image, axis=0)  # Adiciona batch dimension
    return image

def detect_objects(video_path):
    """Detecta objetos no vídeo usando o modelo treinado no TensorFlow."""
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processar frame
        processed_frame = preprocess_frame(frame)

        # Fazer a previsão com o modelo
        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Se a confiança for alta, mostrar o objeto detectado
        if confidence > 0.5:
            label = CLASS_NAMES[predicted_class]
            print(f"Objeto detectado: {label} (Confiança: {confidence:.2f})")

            # Exibir texto no frame
            cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Enviar alerta se for um objeto perigoso
            if label in CLASS_NAMES:  # Ajuste conforme necessário
                send_alert(frame)

        # Exibir frame na tela
        cv2.imshow("Detecção", frame)
        if cv2.waitKey(1) == 27:  # Pressionar ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = os.path.join(VIDEOS_DIR, "video2.mp4")  # Substitua pelo caminho do vídeo a ser analisado
    detect_objects(video_path)

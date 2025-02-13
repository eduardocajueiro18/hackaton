import os

# Configuração do TensorFlow para suprimir avisos desnecessários
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Parâmetros globais
DATASET_DIR = "../dataset/"
MODEL_DIR = "../models/"
IMG_SIZE = (224, 224)  # Reduzido para melhorar eficiência
BATCH_SIZE = 32
NUM_CLASSES = 2

def check_data_folders():
    """Verifica se as pastas do dataset existem e quantifica as imagens."""
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Erro: Pastas 'train' e 'test' não encontradas em '{DATASET_DIR}'.")

    print(f"Imagens na pasta train: {sum([len(files) for _, _, files in os.walk(train_path)])}")
    print(f"Imagens na pasta test: {sum([len(files) for _, _, files in os.walk(test_path)])}")

def load_dataset():
    """Carrega e prepara o dataset para treinamento e validação com otimização."""
    check_data_folders()

    dataset_train = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    ).prefetch(tf.data.AUTOTUNE)  # 🔥 Otimiza o carregamento dos dados

    dataset_val = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)  # 🔥 Otimiza o carregamento dos dados

    return dataset_train, dataset_val


def create_model():
    """Cria e compila o modelo de rede neural convolucional (CNN)."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(416, 416, 3)),  # Definição explícita da entrada

        # Camadas convolucionais
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # **AQUI GARANTIMOS QUE OS DADOS SEJAM ACHATADOS**
        tf.keras.layers.Flatten(),  # Transforma (416, 416, 3) em um vetor 1D

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model):
    """Treina o modelo e salva os arquivos de pesos."""
    dataset_train, dataset_val = load_dataset()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Callbacks para otimização do treinamento
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"), monitor="val_loss", save_best_only=True)

    # Treinamento do modelo
    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=50,
        callbacks=[early_stopping, reduce_lr, model_checkpoint]
    )

    # Salvamento do modelo final
    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model.save(final_model_path)
    print(f"Modelo final salvo em: {final_model_path}")

if __name__ == "__main__":
    model = create_model()
    train_model(model)

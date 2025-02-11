import os

# Configuração do TensorFlow para suprimir avisos desnecessários
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Apenas mostra erros críticos

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Parâmetros globais
DATASET_DIR = "../dataset/"
MODEL_DIR = "../models/"  # Pasta para salvar o modelo
IMG_SIZE = (416, 416)
BATCH_SIZE = 32
NUM_CLASSES = 2  # Ajuste para o número correto de classes

def check_data_folders():
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Erro: As pastas 'train' e 'test' não foram encontradas em '{DATASET_DIR}'.")

    print(f"Imagens na pasta train: {sum([len(files) for _, _, files in os.walk(train_path)])}")
    print(f"Imagens na pasta test: {sum([len(files) for _, _, files in os.walk(test_path)])}")

def check_dataset_path():
    """Verifica se o diretório do dataset existe."""
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Erro: O diretório do dataset '{DATASET_DIR}' não foi encontrado.")

def load_dataset():
    """Carrega e prepara o dataset para treinamento e validação."""
    check_dataset_path()

    dataset_train = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    class_names = dataset_train.class_names
    print(f"Classes detectadas: {class_names}")
    print(f"Quantidade de classes detectadas: {len(class_names)}")

    dataset_val = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False  # Melhor para validação
    )

    return dataset_train, dataset_val

def create_model():
    """Cria e compila o modelo de rede neural convolucional (CNN)."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Definição explícita da entrada

        # Camadas convolucionais para melhor extração de características
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Reduz overfitting
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Ajuste conforme o número de classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model):
    """Treina o modelo e salva os arquivos de pesos."""
    dataset_train, dataset_val = load_dataset()

    # Criar diretório models/ se não existir
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Monitorar Overfitting - Para evitar overfitting sem precisar definir epochs fixo
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Acompanha a perda na validação
        patience=3,  # Para se a perda na validação não melhorar por 3 épocas seguidas
        restore_best_weights=True
    )

    # Treinar o modelo
    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=50,  # Define um número grande, mas EarlyStopping interrompe antes
        callbacks=[early_stopping]
    )

    # Salvar modelo em formato KERAS (.keras)
    keras_path = os.path.join(MODEL_DIR, "model.keras")
    model.save(keras_path)
    print(f"Modelo salvo em: {keras_path}")

if __name__ == "__main__":
    check_data_folders()
    model = create_model()
    train_model(model)

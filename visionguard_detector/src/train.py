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
    """Carrega e prepara o dataset com aumento de dados e normalização."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,            # Normalização
        rotation_range=20,         # Rotação aleatória
        width_shift_range=0.2,     # Deslocamento horizontal
        height_shift_range=0.2,    # Deslocamento vertical
        shear_range=0.2,           # Transformação em cisalhamento
        zoom_range=0.2,            # Zoom aleatório
        horizontal_flip=True       # Espelhamento horizontal
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    dataset_train = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    dataset_val = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print(f"Classes detectadas: {dataset_train.class_indices}")
    return dataset_train, dataset_val

def create_model():
    """Cria um modelo CNN otimizado para classificação de imagens."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # Definição correta da entrada
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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
    check_data_folders()
    model = create_model()
    train_model(model)

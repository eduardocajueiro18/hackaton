import tensorflow as tf

def load_dataset():
    # Carregar e preparar o dataset para treinamento (exemplo fictício)
    pass

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(416, 416, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Ajuste conforme o número de classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model):
    dataset_train, dataset_val = load_dataset() 
    model.fit(dataset_train,
              validation_data=dataset_val,
              epochs=10)  # Ajuste o número de épocas conforme necessário

if __name__ == "__main__":
    model = create_model()
    train_model(model)

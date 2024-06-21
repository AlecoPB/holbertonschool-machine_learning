import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def preprocess_data(X, Y):
    # Normalize the data
    X = X.astype('float32') / 255.0
    # One-hot encode the labels
    Y = to_categorical(Y, num_classes=10)
    return X, Y

def create_model(input_shape):
    # Load the pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add new trainable layers
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=output)
    return model

def main():
    # Load and preprocess CIFAR-10 data
    (X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val)
    
    # Define input shape for the pre-trained model
    input_shape = (224, 224, 3) # ResNet50 input shape
    
    # Resize CIFAR-10 images to match pre-trained model input size
    X_train_resized = tf.image.resize(X_train, [224, 224])
    X_val_resized = tf.image.resize(X_val, [224, 224])
    
    # Create the model
    model = create_model(input_shape)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_resized, Y_train, validation_data=(X_val_resized, Y_val), epochs=10, batch_size=32)
    
    # Save the model
    model.save('cifar10.h5')

if __name__ == '__main__':
    main()

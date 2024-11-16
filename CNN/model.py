import keras
from keras.layers import Conv1D, Dropout, Dense, Flatten, MaxPooling1D
from keras.models import Sequential
from graph_sage_embedding import length_of_embedding


CNN = keras.Sequential()
CNN.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=( length_of_embedding,1)))
CNN.add(keras.layers.MaxPooling1D(pool_size=2))
CNN.add(keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
CNN.add(keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
CNN.add(keras.layers.MaxPooling1D(pool_size=2))
CNN.add(keras.layers.Dropout(0.3))
CNN.add(keras.layers.Flatten())
CNN.add(keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
CNN.add(keras.layers.Dense(2, activation='softmax'))
CNN.compile(loss='categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy', 'precision', 'recall', 'f1_score'])
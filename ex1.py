import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(x_train.shape)

x_train, x_test = x_train/255.0, x_test/255.0

model = models.Sequential([
    #CNN
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.Conv2D(32, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (2, 2), activation='relu'),
    layers.AveragePooling2D((2, 2)),

    layers.Flatten(),#1024units
    
    #Full conected
    layers.Reshape((32, 32)),
    
    #RNN_LSTM
    layers.LSTM(128, input_shape = (32, 32), activation = 'relu', return_sequences = True),
    layers.LSTM(64, activation = 'relu', dropout = 0.1, recurrent_dropout = 0.1),

    #Full connected
    layers.Dense(32, activation = 'relu'),
    #layers.Dropout(rate = 0.1),
    layers.Dense(20, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
    ])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=128)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


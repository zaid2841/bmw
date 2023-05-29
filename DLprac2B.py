# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:15:46 2023

@author: aarti
"""

from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print("Train Shape :",x_train.shape)
print("Test Shape :",x_test.shape)
print("y_train shape :",y_train.shape)
print("y_test shape :",y_test.shape)
print(x_train[1])
print(y_train[1])
vocab=imdb.get_word_index()
print(vocab['the'])
class_names=['Negative', 'Positive']
reverse_index = dict([(value, key) for (key, value) in vocab.items()])
def decode(review):
    text=""
    for i in review:
        text=text+reverse_index[i]
        text=text+" "
    return text
decode(x_train[1])
def showlen():
    print("Length of first training sample: ",len(x_train[0]))
    print("Length of second training sample: ",len(x_train[1]))
    print("Length of first test sample: ",len(x_test[0]))
    print("Length of second test sample: ",len(x_test[1]))
showlen()
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train=pad_sequences(x_train, value=vocab['the'], padding='post', maxlen=256)
x_test=pad_sequences(x_test, value=vocab['the'], padding='post', maxlen=256)
showlen()
decode(x_train[1])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
model=Sequential()
model.add(Embedding(10000,16))
model.add(GlobalAveragePooling1D())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=4, batch_size=128, verbose=1,validation_data=(x_test, y_test))
x_test[10]
y_test[10]

import numpy as np
predicted_value=model.predict(np.expand_dims(x_test[10], 0))
print(predicted_value)
if predicted_value>0.5:
    final_value=1
else:
    final_value=0
print(final_value)
print(class_names[final_value])

loss, accuracy = model.evaluate(x_test, y_test)
print("Loss :",loss)
print("Accuracy (Test Data) :",accuracy*100)
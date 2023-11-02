# -*- coding: utf-8 -*-
#transfer learning with natural language

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

#%%  Download Data
import requests
import pandas as pd

# URL of the file to download
url = "https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P"

# Specify the file path where you want to save the downloaded file
file_path = r'D:\Weiter_Bildung\Machine learning\sentiment.csv'

# Download the file using requests
response = requests.get(url)
with open(file_path, 'wb') as file:
    file.write(response.content)

# Read the CSV file using pandas
df = pd.read_csv(file_path)


#%%  Split Data
X = df['text']
y = df['sentiment']
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)

#%%  Tokenizer
vocab_size = 10000
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)

#%%

word_index = tokenizer.word_index

#%%  convert to vector

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
#%%

padding_type='post'
truncation_type='post'
max_length = 100

X_train_padded = pad_sequences(X_train_sequences, padding=padding_type, 
                               truncating= truncation_type, maxlen=max_length)
 
X_test_padded = pad_sequences(X_test_sequences, padding=padding_type, 
                               truncating= truncation_type, maxlen=max_length)

#%%  Using GloVe Embeddings

import requests

# URL of the GloVe embeddings file
url = "http://nlp.stanford.edu/data/glove.6B.zip"

# Send a GET request to the URL
response = requests.get(url, stream=True)

# Save the content of the response to a local file
with open(r'D:\Weiter_Bildung\Machine learning\glove.6B.zip', 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

    
#%%  Next, extract them into a temporary folder.
# with zipfile.ZipFile(r'D:\Weiter_Bildung\Machine learning\glove.6B.zip', 'r') as zip_ref:
    # zip_ref.extractall(r'D:\Weiter_Bildung\Machine learning')
    
#%%  Load the Glove embeddings, and append them to a dictionary.
import numpy as np

embeddings_index = {}
glove_file_path = r'D:\Weiter_Bildung\Machine learning\glove.6B.100d.txt'

with open(glove_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


#%%   Use this dictionary to create an embedding matrix for each word in the training se
embedding_matrix = np.zeros((len(word_index) + 1, max_length))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#%% In case a word isn’t found, zero will represent it. For example, here is the embedding vector for the word bakery.
embeddings_index.get("bakery")

#%%  Create the embedding layer

embedding_layer = Embedding(len(word_index) + 1,
                            max_length,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)

#%%   Create the model 

model = Sequential([
    embedding_layer,
    Bidirectional(LSTM(150, return_sequences=True)),
    Bidirectional(LSTM(150)),
    Dense(6, activation='relu'),
   Dense(1, activation='sigmoid')
])

#%% Training the model

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#%%  EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#%%

# %load_ext tensorboard
# rm -rf logs
log_folder = 'logs'
callbacks = [
            EarlyStopping(patience = 10, monitor="val_loss"),
            TensorBoard(log_dir=log_folder)
            ]
num_epochs = 600
history = model.fit(X_train_padded, y_train, epochs=num_epochs, validation_data=(X_test_padded, y_test),callbacks=callbacks)

#%%   The performance of the model can be e using the `evaluate` function.

loss, accuracy = model.evaluate(X_test_padded,y_test)
print('Test accuracy :', accuracy)


























import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

def load_and_preprocess_text(filepath, max_lines=10000):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.readlines()[:max_lines]  # Load only the first max_lines lines
    text = [line.lower() for line in text if line.strip()]  # Convert to lowercase and remove empty lines
    return text

def train_model_on_dataset(text_lines, batch_size=1000):
    tokenizer = Tokenizer(num_words=5000)  
    total_words = 5000
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=None))  # Embedding layer initialization
    max_sequence_len = 0

    for i in range(0, len(text_lines), batch_size):
        batch_lines = text_lines[i:i+batch_size]
        tokenizer.fit_on_texts(batch_lines)
        
        input_sequences = []
        for line in batch_lines:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for j in range(1, len(token_list)):
                n_gram_sequence = token_list[:j+1]
                input_sequences.append(n_gram_sequence)
        
        max_sequence_len = max(max_sequence_len, max([len(x) for x in input_sequences]))

        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        X, y = input_sequences[:,:-1], input_sequences[:,-1]
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        
        if i == 0:  # Compile the model after the first batch
            model.add(LSTM(100))  # LSTM layer initialization
            model.add(Dense(total_words, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model in batches
        model.fit(X, y, epochs=3, verbose=1)

    return model, tokenizer, max_sequence_len

def fine_tune_model(model, tokenizer, user_texts, max_sequence_len):
    tokenizer.num_words = 5000  
    tokenizer.fit_on_texts(user_texts)
    
    input_sequences = []
    for line in user_texts:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for j in range(1, len(token_list)):
            n_gram_sequence = token_list[:j+1]
            input_sequences.append(n_gram_sequence)

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=5000)  # Use the same number of classes

    # Fine-tune the model
    model.fit(X, y, epochs=3, verbose=1)  # Fine-tune with fewer epochs

    return model

def predict_next_word(model, tokenizer, max_sequence_len, seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""

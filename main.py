#Import the libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()

# Reverse the word index to decode reviews
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pretrained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Helper function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])   

# Function to preprocess the user input
def preprocess_input(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is the index for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## prediction function
def predict_review(text):
    preprocessed_review = preprocess_input(text)

    prediction = model.predict(preprocessed_review)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]


# Streamlit app
import streamlit as st

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# User input
user_input = st.text_area("Enter your movie review:")

if st.button("Classify"):
    sentiment, score = predict_review(user_input)

    st.write(f'Review: "{user_input}"')
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score:.4f}')

else:
    st.write("Please enter a review and click 'Classify' to see the sentiment analysis.")

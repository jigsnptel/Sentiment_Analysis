from engine import SentimentAnalyzer
import streamlit as st


# Load the sentiment analysis model from Hugging Face
sentiment_analysis = SentimentAnalyzer()

# Define the Streamlit app interface
st.title("User Sentiment Analysis")

sentence = st.text_input("Enter a sentence:")

# Perform sentiment analysis on the input sentence
if sentence:
    label = sentiment_analysis.get_sentiment(sentence)
    # Display the sentiment analysis result to the user
    st.write(f"Sentiment analysis result: {label}")
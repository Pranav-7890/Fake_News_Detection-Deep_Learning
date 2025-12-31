import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# --- 1. CLEANING FUNCTION ---
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# --- 2. LOAD MODEL & TOKENIZER ---
# --- 2. LOAD MODEL & TOKENIZER ---
@st.cache_resource 
def load_all_assets():
    base_path = os.path.dirname(__file__)
    
    # 1. Build the clean skeleton
    model = Sequential([
        tf.keras.layers.Input(shape=(300,)), 
        Embedding(10000, 16),
        Bidirectional(LSTM(32)), 
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.build(input_shape=(None, 300)) 
    
    # 2. Point to your EXISTING .h5 file
    h5_path = os.path.join(base_path, 'fake_news_detector.keras')
    
    # 3. Load ONLY the weights from that .h5 file
    # skip_mismatch=True helps ignore those pesky hidden config errors
    model.load_weights(h5_path, skip_mismatch=True)
    
    # Load tokenizer
    tok_path = os.path.join(base_path, 'tokenizer.pickle')
    with open(tok_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

model, tokenizer = load_all_assets()

# --- 3. USER INTERFACE ---
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to see if our AI thinks it's **Real** or **Fake**.")

user_input = st.text_area("Paste News Article Here:", height=200)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner('Analyzing...'):
            # Process input
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=300, padding='post', truncating='post')
            
            # Predict
            prediction = model.predict(padded)[0][0]
            real_score = float(prediction)
            fake_score = 1.0 - real_score

            # Display Results
            st.divider()
            if real_score > 0.5:
                st.success(f"### This looks like REAL NEWS")
            else:
                st.error(f"### This looks like FAKE NEWS")
            
            # Show Confidence Bars
            st.write(f"Confidence (Real): {real_score:.2%}")
            st.progress(real_score)
            st.write(f"Confidence (Fake): {fake_score:.2%}")
            st.progress(fake_score)
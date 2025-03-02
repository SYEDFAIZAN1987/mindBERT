import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Define model and tokenizer paths from Hugging Face
MODEL_PATH = "DrSyedFaizan/mindBERT"

# Load tokenizer and model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Streamlit UI setup
st.title("MindBERT - Mental Health Analysis Chat")
st.write("Enter a message, and the model will analyze the mental state of the writer.")

user_input = st.text_area("Type your message here:")

if st.button("Analyze Mental State"):
    if user_input.strip():
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Mapping predicted class to mental state
        label_map = {
            0: "Anxiety",
            1: "Bipolar",
            2: "Depression",
            3: "Normal",
            4: "Personality Disorder",
            5: "Stress",
            6: "Suicidal"
        }
        mental_state = label_map.get(predicted_class, "Unknown")
        
        # Display results
        st.write(f"Predicted Mental State: **{mental_state}**")
    else:
        st.warning("Please enter some text for analysis.")

# Footer
st.markdown("---")
st.markdown("Developed by Dr. Syed Faizan using MindBERT on Hugging Face.")

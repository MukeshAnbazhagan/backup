import streamlit as st
import torchaudio
import torch
import tempfile
import os
import time  # Added time module for timing
import re  # Added for word counting
from transformers import AutoModel
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

st.set_page_config(page_title="Speech to Text and Translation", layout="wide")
st.title("Audio Transcription and Translation")

@st.cache_resource
def load_transcription_model():
    try:
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", 
            trust_remote_code=True,
            device="cuda:1" 
        )
        return model
    except Exception as e:
        st.error(f"Error loading transcription model: {str(e)}")
        return None

@st.cache_resource
def load_translation_model():
    try:
        llm = ChatOllama(
            model="gemma3:27b",
            base_url="http://localhost:11434",
            temperature=0.2,
            device="cuda:0" 
        )
        return llm
    except Exception as e:
        st.error(f"Error loading translation model: {str(e)}")
        return None

def count_words(text):
    """Count the number of words in text by splitting on whitespace"""
    if not text:
        return 0
    # Split on whitespace and filter out empty strings
    words = [word for word in re.split(r'\s+', text) if word]
    return len(words)

def transcribe_audio(file_path: str, model, lang: str = "ta"):
    try:
        wav, sr = torchaudio.load(file_path)
        wav = torch.mean(wav, dim=0, keepdim=True)  # Convert stereo to mono if needed
     
        # Resample to 16kHz if required
        target_sample_rate = 16000
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            wav = resampler(wav)
     
        # Run CTC decoding
        transcription_ctc = model(wav, lang, "ctc")
        return transcription_ctc
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def translate_tamil_to_english(tamil_text, llm):
    """
    Translate Tamil text to English
    """
    if not tamil_text:
        return ""
        
    prompt = f"""
    Translate the following Tamil text to English:
    - Return only the translation without any additional explanations
    - Preserve proper nouns as they are
    
    Tamil text: {tamil_text}
    English translation:
    """.strip()
    
    try:
        response = llm([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return tamil_text  # Return original if translation fails

# Sidebar for model loading status
with st.sidebar:
    st.header("Model Status")
    
    transcription_model_status = st.empty()
    translation_model_status = st.empty()
    
    with transcription_model_status.container():
        st.info("Loading transcription model...")
        transcription_model = load_transcription_model()
        if transcription_model is not None:
            st.success("✅ Transcription model loaded successfully")
        else:
            st.error("❌ Failed to load transcription model")
    
    with translation_model_status.container():
        st.info("Loading translation model...")
        translation_model = load_translation_model()
        if translation_model is not None:
            st.success("✅ Translation model loaded successfully")
        else:
            st.error("❌ Failed to load translation model")

# Main content area
st.subheader("Upload Tamil Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

col1, col2 = st.columns(2)

tamil_text = ""
english_text = ""
transcription_time = None
translation_time = None

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    
    st.audio(uploaded_file, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
    
    if st.button("Transcribe and Translate"):
        with st.spinner("Transcribing audio..."):
            if transcription_model is not None:
                start_time = time.time()  # Start timing for transcription
                tamil_text = transcribe_audio(temp_file_path, transcription_model)
                end_time = time.time()  # End timing for transcription
                transcription_time = end_time - start_time  # Calculate transcription time
                
                if tamil_text:
                    # Count words in Tamil text
                    tamil_word_count = count_words(tamil_text)
                    
                    with col1:
                        st.subheader("Transcription")
                        st.text_area("", value=tamil_text, height=300, disabled=True)
                        # Display transcription time and word count
                        st.info(f"Transcription completed in {transcription_time:.2f} seconds | Word count: {tamil_word_count}")
                else:
                    st.error("Failed to transcribe audio.")
            else:
                st.error("Transcription model not loaded. Please check the sidebar for errors.")
        
        with st.spinner("Translating to English..."):
            if translation_model is not None and tamil_text:
                start_time = time.time()  # Start timing for translation
                english_text = translate_tamil_to_english(tamil_text, translation_model)
                end_time = time.time()  # End timing for translation
                translation_time = end_time - start_time  # Calculate translation time
                
                if english_text:
                    # Count words in English text
                    english_word_count = count_words(english_text)
                    
                    with col2:
                        st.subheader("English Translation")
                        st.text_area("", value=english_text, height=300, disabled=True)
                        # Display translation time and word count
                        st.info(f"Translation completed in {translation_time:.2f} seconds | Word count: {english_word_count}")
                else:
                    st.error("Failed to translate text.")
            elif tamil_text:
                st.error("Translation model not loaded. Please check the sidebar for errors.")
        
        # Add total processing time if both processes completed successfully
        if transcription_time is not None and translation_time is not None:
            # Get the word counts if available
            tamil_word_count = count_words(tamil_text) if tamil_text else 0
            english_word_count = count_words(english_text) if english_text else 0
            
            total_time = transcription_time + translation_time
            st.success(f"Total processing time: {total_time:.2f} seconds | Tamil words: {tamil_word_count} | English words: {english_word_count}")
    
    # Clean up the temporary file
    try:
        os.unlink(temp_file_path)
    except:
        pass

# Footer
st.markdown("---")
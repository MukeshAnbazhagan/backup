import streamlit as st
import tempfile
import time  # Added time module for timing
import re  # Added for word counting
# NVIDIA NeMo ASR
import nemo.collections.asr as nemo_asr

@st.cache_resource(show_spinner=True)
def load_model():
    """
    Load and cache the ASR model once.
    """
    return nemo_asr.models.ASRModel.from_pretrained(    
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

def count_words(text):
    """Count the number of words in text by splitting on whitespace"""
    if not text:
        return 0
    # Split on whitespace and filter out empty strings
    words = [word for word in re.split(r'\s+', text) if word]
    return len(words)

# Page configuration
st.set_page_config(
    page_title="Nemo ASR Transcription",
    page_icon="🎤",
    layout="centered",
)

st.title("Audio Transcription with NVIDIA NeMo ASR")
st.write(
    "Upload an audio file (WAV/MP3/OGG) and click Transcribe to see the transcription."
)

# File uploader
uploaded_file = st.file_uploader(
    label="Choose an audio file",
    type=["wav", "mp3", "ogg", "flac"],
)

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(suffix="." + uploaded_file.name.split('.')[-1], delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    # Transcribe button
    if st.button("Transcribe"):
        asr_model = load_model()
        with st.spinner("Transcribing..."):
            try:
                start_time = time.time()  # Start timing
                result = asr_model.transcribe([tmp_path])
                transcription = result[0].text
                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Calculate elapsed time
            except Exception as e:
                st.error(f"Error during transcription: {e}")
                transcription = None
                elapsed_time = None
        
        if transcription:
            st.subheader("Transcription")
            st.write(transcription)
            
            # Count words in the transcription
            word_count = count_words(transcription)
            
            # Display the timing information and word count
            if elapsed_time is not None:
                st.info(f"Transcription completed in {elapsed_time:.2f} seconds | Word count: {word_count}")
                
            # Add separate metrics display
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Processing Time", value=f"{elapsed_time:.2f} sec")
            with col2:
                st.metric(label="Word Count", value=word_count)
        else:
            st.write("No transcription available.")
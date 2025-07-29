from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv
from sarvamai import SarvamAI
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv(override=True)



SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    logger.error("SARVAM_API_KEY environment variable is not set")
    raise EnvironmentError("SARVAM_API_KEY environment variable is not set")

# Initialize Sarvam AI client
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required for LLM processing")

# Configuration
MODEL_NAME = "gpt-4o-2024-11-20"

app = FastAPI(title="Audio Form Data Extractor", description="Upload audio and extract form data")

universal_form_prompt = """
You are an expert transcription-and-extract assistant.  
The audio may be in any Indian language or Indian-accented English.  
Your tasks:

1. Transcribe **verbatim** in the original script—do **not** translate.  
2. Whenever the speaker reads out a form field (from the list below), **normalize** it into the correct format:
   - firstName: capitalize (e.g. "राहुल" → "Rahul")  
   - lastName: capitalize  
   - birthDay: extract integer 1-31  
   - birthMonth: extract integer 1-12  
   - birthYear: extract 4-digit year  
   - gender: one of "Male", "Female", "Other"  
   - state: English state name (capitalize, e.g. "Maharashtra")  
   - district: English district name  
   - areaType: "Urban" or "Rural"  
   - youthType: one of "NBS", "NCC", "NYKS", "BSG", "Others"  
   - sportsTalent: the sport's name (capitalize)  
   - kheloIndia: boolean true if they say "yes" or "participated", false if "no" or "did not participate"

3. **Keep** all other speech exactly as read—no hallucinations, no extra corrections. 
"""

class YouthFormData(BaseModel):
    firstName: Optional[str] = Field(default="Unknown", description="First name of the applicant")
    lastName: Optional[str] = Field(default="Unknown", description="Last name or surname of the applicant")
    birthDay: Optional[int] = Field(default=1, description="Day of birth (1-31)")
    birthMonth: Optional[int] = Field(default=1, description="Month of birth (1-12)")
    birthYear: Optional[int] = Field(default=2000, description="Year of birth (YYYY)")
    gender: Optional[Literal["Male", "Female", "Other"]] = Field(default="Other", description="Gender of the applicant")
    state: Optional[str] = Field(default="Unknown", description="State name in English")
    district: Optional[str] = Field(default="Unknown", description="District name in English")
    areaType: Optional[Literal["Urban", "Rural"]] = Field(default="Urban", description="Urban or Rural area")
    youthType: Optional[Literal["NBS", "NCC", "NYKS", "BSG", "Others"]] = Field(default="Others", description="Youth category type")
    sportsTalent: Optional[str] = Field(default="None", description="Sport the applicant is talented in")
    kheloIndia: Optional[bool] = Field(default=False, description="Whether the applicant participated in Khelo India (true/false)")

class TranscriptionResponse(BaseModel):
    transcript: str
    form_data: YouthFormData
    success: bool
    message: str


extraction_prompt = PromptTemplate(
    template="""
    You are a multilingual assistant that extracts structured form information from the following transcript (in any Indian language).

    Instructions:
    - If a field is missing or unclear, use the default values shown below.
    - Normalize fields according to the expected format.

    Expected Fields & Defaults:
    - firstName: capitalize (default: "Unknown")
    - lastName: capitalize (default: "Unknown")
    - birthDay: integer (1-31) (default: 1)
    - birthMonth: integer (1-12) (default: 1)
    - birthYear: 4-digit year (default: 2000)
    - gender: "Male"/"Female"/"Other" (default: "Other")
    - state: English state name (default: "Unknown")
    - district: English district name (default: "Unknown")
    - areaType: "Urban"/"Rural" (default: "Urban")
    - youthType: "NBS"/"NCC"/"NYKS"/"BSG"/"Others" (default: "Others")
    - sportsTalent: sport name or "None"
    - kheloIndia: true if participated, false otherwise

    Correct spoken words like:
    - "at the rate" → "@"
    - "dot" → "."
    - "dash" → "-"

    Transcript:
    \"\"\"{text}\"\"\"

    Output only the JSON object.
    """, 
    input_variables=["text"]
)

def convert_webm_to_wav(input_path: str) -> str:
    """Convert WebM audio to WAV format for Sarvam AI compatibility."""
    output_path = input_path.replace('.webm', '.wav')
    try:
        # Use ffmpeg to convert WebM to WAV
        subprocess.run([
            'ffmpeg', '-i', input_path, 
            '-acodec', 'pcm_s16le', 
            '-ar', '16000', 
            '-ac', '1', 
            output_path, '-y'
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found. Please install ffmpeg.")

def transcribe_audio(audio_file_path: str) -> str:
    """Run transcription using Sarvam AI and return raw result text."""
    try:
        # Convert to WAV if it's WebM
        if audio_file_path.endswith('.webm'):
            audio_file_path = convert_webm_to_wav(audio_file_path)
        
        response = sarvam_client.speech_to_text.translate(
            file=open(audio_file_path, "rb"),
            model="saaras:v2"
        )
        return response.transcript
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

def extract_form_data(transcript: str) -> dict:
    """Extract structured form data using the LLM chain."""
    try:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=api_key)
        chain = extraction_prompt | llm.with_structured_output(YouthFormData)
        result = chain.invoke({"text": transcript})
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form extraction failed: {str(e)}")

@app.post("/process-audio/", response_model=TranscriptionResponse)
async def process_audio(audio_file: UploadFile = File(...)):
    """Process uploaded audio file and extract form data."""
    
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
        content = await audio_file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Transcribe the audio using Sarvam AI
        transcript = transcribe_audio(tmp_file_path)
        
        # Extract form data
        form_data = extract_form_data(transcript)
        
        return TranscriptionResponse(
            transcript=transcript,
            form_data=YouthFormData(**form_data),
            success=True,
            message="Audio processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Audio Form Data Extractor is running"}

async def main():
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=8001)
    server = uvicorn.Server(config)
    await server.serve()


# Add these imports at the top of your FastAPI file
from fastapi.middleware.cors import CORSMiddleware

# Add this after creating your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
        else:
            raise
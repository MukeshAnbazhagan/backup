# MyBharat Voice Registration System

Voice-to-form application that converts speech in Indian languages to structured form data.

## Features
- Record voice in Hindi, Tamil, Telugu, English and other Indian languages
- Auto-fill registration form from speech
- Toggle recording (press to start/stop)
- Real-time processing with AI

## Requirements
- Python 3.8+
- FFmpeg
- Sarvam AI API Key
- OpenAI API Key

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
sudo apt install ffmpeg  # Linux
```

2. Create `.env` file:
```bash
SARVAM_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

3. Run backend:
```bash
python fastapi_server.py
```

4.Run backend:
```bash
python3 -m http.server 80511
```
5.Open `mybharat_form.html` in browser

## Usage
1. Click microphone button
2. Speak your details in any Indian language
3. Click microphone again to stop
4. Form auto-fills with extracted data
5. Review and submit

## Example Voice Input
"My name is Priya Sharma, born 15th March 1999, female, from Mumbai Maharashtra, NCC cadet, cricket player"

## Files
- `fastapi_server.py` - Backend API
- `mybharat_form.html` - Frontend form
- `requirements.txt` - Python packages
- `.env` - API keys


## Refer to Vice_to_text.pdf to access the weburl
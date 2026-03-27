# Audio Bot RAG

This project is a lightweight Streamlit app for a retrieval-augmented audio bot using Whisper, GPT, and gTTS.

## Clone the repository

```bash
git clone https://github.com/mrdavidlong/audio-bot-rag.git
cd audio-bot-rag
```

## One-time setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file from the example:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder with your real OpenAI API key:

```env
OPENAI_API_KEY=your_real_openai_api_key
```

OR, set the key directly in your terminal session before launching the app:

```bash
export OPENAI_API_KEY="your_real_openai_api_key"
```

## Run the app

Start the Streamlit app:

```bash
streamlit run audio_app_rag_lite.py
```

Then open the app in your browser at:

```text
http://localhost:8501
```

## Notes

- The app uses your microphone through Streamlit's audio input widget, so allow browser microphone access when prompted.
- On first run, the app creates local knowledge base files in the project directory.
- Keep your virtual environment activated while running the app.
- The app loads environment variables from `.env`.

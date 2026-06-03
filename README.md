# WorldScribe

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure secrets. Copy the example env file and fill in your own keys:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and provide:
   - `OPENAI_API_KEY` — your OpenAI API key
   - `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` — your Azure Speech credentials
   - `SERVER_IP` — IP of the machine running the visual server

3. Firebase: download your own Firebase Admin SDK service-account JSON from the
   Firebase console and place it in the project root. This file is gitignored and
   must never be committed.

> Note: `.env`, credential JSON files, images, and other data/media files are
> excluded from version control via `.gitignore`.

# WorldScribe

> ⚠️ **Work in progress — under construction.** This repository is still being
> developed and is **not yet ready for use**. The code, setup instructions, and
> documentation are incomplete and may change or break at any time. Use at your own
> risk.

WorldScribe is a system that generates **live, context-aware visual descriptions** of
a user's surroundings in real time. A smartphone streams the camera feed to a local
machine, where frames are processed with object detection and vision-language models
(and optionally GPT-4V) to produce descriptions that are spoken back to the user via
text-to-speech. It is designed as an assistive tool to help people understand their
visual environment on the go.

## Architecture

```
 Smartphone  ──TCP──►  Local laptop (main.py)  ──►  Firebase (Realtime DB / Firestore)
  (camera)              - object detection (YOLO)         ▲
                        - frame display / debug           │
                                                          ▼
                                            GPU server (worldscribe_remote_server.py)
                                            - moondream2 VLM + GPT-4V captioning
                                            - hand / gesture understanding
                                            - TTS (OpenAI / Azure) output
```

- **`main.py`** — runs on the local laptop. Receives the camera stream over TCP, runs
  object detection, and coordinates with Firebase.
- **`worldscribe_remote_server.py` / `remote_server.py`** — run on a GPU machine. Host
  the vision-language models, perform captioning/ranking, and drive speech output.
- **`utils/`** — captioning (`gpt4v.py`), speech (`speech.py`, `speech_azure.py`),
  gesture/hand understanding (`hands23/`, `gesture/`), Firebase managers
  (`firebase/`), NLP post-processing, and more.
- **`similarity/`** — image/frame similarity models used to decide when the scene has
  changed enough to warrant a new description.
- **`config/`** — server IP/port configuration.

## Requirements

- Python 3.8+
- A CUDA-capable GPU for the model server (`worldscribe_remote_server.py`)
- A Firebase project (Realtime Database + Firestore)
- An OpenAI API key and/or Azure Speech credentials for captioning and TTS

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure secrets. Copy the example env file and fill in your own values:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env`:
   - `OPENAI_API_KEY` — your OpenAI API key
   - `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` — your Azure Speech credentials
   - `SERVER_IP` — IP of the machine running the visual server
   - `FIREBASE_CREDENTIALS` — path to your Firebase Admin SDK service-account JSON
   - `FIREBASE_DATABASE_URL` — your Firebase Realtime Database URL

3. Firebase credentials. Download your own Firebase Admin SDK service-account JSON
   from the Firebase console and place it in the project root, then set
   `FIREBASE_CREDENTIALS` and `FIREBASE_DATABASE_URL` in `.env`. The JSON file is
   gitignored and **must never be committed**. All Firebase identifiers are read from
   the environment via `utils/env_config.py` — no project names, file names, or URLs
   are hardcoded in the source.

4. Server config. Set the server IP/port in `config/config_server.py` (or via the
   `SERVER_IP` environment variable).

## Usage

On the local laptop (camera ingestion + object detection):

```bash
python main.py
```

On the GPU server (vision-language models + captioning + TTS):

```bash
python worldscribe_remote_server.py
```

Then connect the smartphone client to the laptop's IP and port to begin streaming.

## Notes on data & privacy

`.env`, Firebase credential JSON files, images, audio, point clouds, and other
data/media files are excluded from version control via `.gitignore`. Provide your own
credentials and assets locally.

## Citation

If you use this work, please cite the WorldScribe paper:

```bibtex
@inproceedings{chang2024worldscribe,
  title     = {WorldScribe: Towards Context-Aware Live Visual Descriptions},
  author    = {Chang, Ruei-Che and others},
  booktitle = {Proceedings of the ACM Symposium on User Interface Software and Technology (UIST)},
  year      = {2024}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for
details.

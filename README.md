# Discord Voice Note API (discord-vn-api)

Lightweight FastAPI backend + static frontend for sending Discord voice notes via Discord voice message attachment API.

## Overview

- `main.py`: FastAPI application with `/voice-notes` endpoint.
- `requirements.txt`: dependencies.
- `static/index.html`: minimal form for clients (TCL Flip 3 friendly).
- `Dockerfile`: for containerized deployment on Back4App.

## Features

- Accepts audio via multipart/form-data POST to `/voice-notes`.
- Converts audio to OGG/Opus (48kHz, mono, 32kbps) using PyAV.
- Sends voice messages to Discord channels via Discord REST API.
- Supports two channels (Channel 1 & Channel 2) via radio buttons.
- Frontend works on low-end devices (TCL Flip 3) with D-Pad navigation.
- Includes visual feedback (flashing red indicator, status messages).
- Frontend hosted on GitHub Pages, backend on Back4App (Docker).

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (create `.env` file):
   ```env
   BOT_TOKEN=your_bot_token_here
   CHANNEL_ID_1=first_channel_id
   CHANNEL_ID_2=second_channel_id
   ```

3. **Run the backend**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Run the frontend server** (for local testing):
   ```bash
   cd docs
   python3 -m http.server 8080
   ```

## API Endpoints

- **POST** `/voice-notes` (multipart/form-data)
  - `audio`: audio file (MP3/WAV/OGG)
  - `channel`: optional channel selector (1 or 2)

  **Response**:
  ```json
  {
    "status": "success",
    "duration_secs": <audio_duration>,
    "file_size_bytes": <file_size>
  }
  ```

- **GET** `/health` (debug endpoint)
  - Returns: `{"status": "ok"}`

## Deployment

- **Backend**: Deploy to Back4App (Docker) - automatic builds from GitHub.
- **Frontend**: Deploy to GitHub Pages (static folder).

## Debugging API

- **POST** `/_debug_waveform` - Upload an audio file to get the generated waveform data.
  ```bash
  curl -X POST http://localhost:8000/_debug_waveform \
    -F "audio=@./your-audio.ogg"
  ```
  Returns:
  ```json
  {
    "waveform_base64": "...",
    "raw_values": [0, 128, 255, ...],
    "length": 256,
    "min": 0,
    "max": 255
  }
  ```

## Notes

- Requires a bot token with `Send Messages` permission in the target channel.
- For local testing, use `curl` or Postman to test the API.
- Frontend uses D-Pad friendly UI with large touch targets.

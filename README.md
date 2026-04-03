# Discord Voice Note API (discord-vn-api)

Lightweight FastAPI backend + static frontend for sending Discord voice notes via Discord voice message attachment API.

## Overview

- `main.py`: FastAPI application with `/send-voice` and `/health`.
- `requirements.txt`: dependencies.
- `static/index.html`: minimal form for clients (TCL Flip 3 friendly).

## Setup

1. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Set env vars:

   ```bash
   export BOT_TOKEN="<your-bot-token>"
   export CHANNEL_ID="<your-discord-channel-id>"
   ```

3. Run:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. Use frontend:

   - Open `static/index.html` on GitHub Pages or local file
   - Enter your backend endpoint, e.g. `https://your-host/send-voice`
   - Select an audio file (MP3/WAV/OGG)
   - Click `Send Voice Note`

### Using a .env file

You can store `BOT_TOKEN` and `CHANNEL_ID` in a `.env` file at the project root to avoid exporting them each shell session. Example `.env`:

```
BOT_TOKEN=your_bot_token_here
CHANNEL_ID_1=channel_id_1
CHANNEL_ID_1=channel_id_2
```

The application uses `python-dotenv` to load these values automatically on startup. After creating or editing `.env`, restart `uvicorn` so the new values are picked up.

## API

POST `/send-voice`
- multipart/form-data `audio` file
- response:
  - `status` (success or error)
  - `duration_secs`
  - `file_size_bytes`

GET `/health`
- returns `{ "status":"ok" }`

## Discord voice message contract

- uploads to `/channels/{CHANNEL_ID}/attachments`
- PUT file to `upload_url`
- POST `/channels/{CHANNEL_ID}/messages` with `flags=8192` and voice attachment metadata

## Notes

- No authorization at API or frontend.
- Uses PyAV to transcode to mono 48kHz Opus in OGG container.
- Generates waveform (<=256 samples) and duration for Discord voice message.

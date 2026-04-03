from fastapi import FastAPI, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from httpx import RequestError
import av
from dotenv import load_dotenv
import io
import os
import base64
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHANNEL_ID_1 = os.environ.get("CHANNEL_ID_1", os.environ.get("CHANNEL_ID", "")).strip()
CHANNEL_ID_2 = os.environ.get("CHANNEL_ID_2", "").strip()

if not BOT_TOKEN or not CHANNEL_ID_1:
    raise RuntimeError("Set BOT_TOKEN and CHANNEL_ID_1 environment variables (you can place them in a .env file)")

DISCORD_API = "https://discord.com/api/v10"
HEADERS = {"Authorization": f"Bot {BOT_TOKEN}"}


def validate_bot_token():
    """Quick check: call Discord /users/@me to validate the bot token."""
    try:
        r = httpx.get(f"{DISCORD_API}/users/@me", headers=HEADERS, timeout=10)
    except RequestError as e:
        raise HTTPException(status_code=502, detail=f"Discord unreachable: {e}")
    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="Discord authentication failed (401). Check BOT_TOKEN.")
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Discord /users/@me error: {r.status_code} {r.text}")
    return r.json()


def convert_to_opus_ogg(input_bytes: bytes) -> bytes:
    """Convert audio to OGG/Opus (48kHz, mono, 32kbps)."""
    input_buf = io.BytesIO(input_bytes)
    output_buf = io.BytesIO()

    with av.open(input_buf) as in_file:
        in_stream = next((s for s in in_file.streams if s.type == "audio"), None)
        if in_stream is None:
            raise ValueError("No audio stream found")

        with av.open(output_buf, 'w', format='ogg') as out_file:
            out_stream = out_file.add_stream('libopus', rate=48000)
            out_stream.layout = 'mono'
            out_stream.bit_rate = 32000

            resampler = av.AudioResampler(format='s16', layout='mono', rate=48000)
            pts_counter = 0
            
            for frame in in_file.decode(audio=0):
                try:
                    resampled = resampler.resample(frame)
                except Exception:
                    resampled = frame

                if resampled is None:
                    continue

                if isinstance(resampled, (list, tuple)):
                    for r in resampled:
                        r.pts = pts_counter
                        pts_counter += r.samples
                        for packet in out_stream.encode(r):
                            out_file.mux(packet)
                else:
                    resampled.pts = pts_counter
                    pts_counter += resampled.samples
                    for packet in out_stream.encode(resampled):
                        out_file.mux(packet)

            for packet in out_stream.encode(None):
                out_file.mux(packet)

    output_buf.seek(0)
    return output_buf.read()


def get_duration(audio_bytes: bytes) -> int:
    """Get duration in seconds from an OGG/Opus file."""
    with av.open(io.BytesIO(audio_bytes)) as container:
        audio_stream = container.streams.audio[0]
        # duration in seconds via time_base and frame count
        if audio_stream.duration is not None and audio_stream.time_base is not None:
            duration = float(audio_stream.duration * audio_stream.time_base)
            return max(1, int(round(duration)))

        # fallback counting frames
        total = 0.0
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                total += float(frame.samples) / frame.sample_rate
        return max(1, int(round(total)))


def generate_waveform(audio_bytes: bytes) -> str:
    """Generate waveform using High-Gain Linear Scaling and a Hard Cap."""
    all_samples = []
    with av.open(io.BytesIO(audio_bytes)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            all_samples.append(frame.to_ndarray().flatten().astype(np.float32))

    if not all_samples:
        return base64.b64encode(bytes([0] * 256)).decode('ascii')

    full_audio = np.abs(np.concatenate(all_samples))
    
    # Use Mean + Std Dev instead of Max. 
    # This naturally ignores peaks and focuses on the 'body' of the sound.
    avg_volume = np.mean(full_audio)
    std_volume = np.std(full_audio)
    reference = avg_volume + (2 * std_volume) or 1.0
    
    raw_chunks = np.array_split(full_audio, 256)
    waveform = []
    
    for chunk in raw_chunks:
        if len(chunk) == 0:
            waveform.append(0)
            continue
            
        # Use Mean Absolute Deviation for the chunk (smoother than Peak)
        val = np.mean(np.abs(chunk))
        
        # 1. HIGH GAIN: Multiply by a large factor to bring small bars up.
        # 2. HARD CAP: Clip anything that goes over the limit.
        # This forces the tall bars to stay at a consistent 'ceiling' 
        # while small bars get pulled up significantly.
        scaled = (val / reference) * 400 
        
        # Add a base height so even 'quiet' chunks have a visible bar
        final_val = int(scaled) + 20 
        
        # Clip to Discord's visual 'sweet spot' (usually around 200-225)
        waveform.append(np.clip(final_val, 0, 225))

    # Strip only literal zeros
    while len(waveform) > 1 and waveform[-1] == 0:
        waveform.pop()

    return base64.b64encode(bytes(waveform)).decode('ascii')

@app.post("/voice-notes")
async def create_voice_note(audio: UploadFile, channel: str = Form(default="1")):
    """Create a voice note and send it to a Discord channel."""
    try:
        # Select channel ID based on form input
        target_channel = CHANNEL_ID_1 if channel == "1" else (CHANNEL_ID_2 or CHANNEL_ID_1)
        
        audio_data = await audio.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio provided")

        ogg_data = convert_to_opus_ogg(audio_data)
        duration = get_duration(ogg_data)
        waveform = generate_waveform(ogg_data)

        # validate token early to provide clearer errors
        validate_bot_token()

        resp = httpx.post(
            f"{DISCORD_API}/channels/{target_channel}/attachments",
            json={
                "files": [
                    {
                        "id": "0",
                        "filename": "voice-message.ogg",
                        "file_size": len(ogg_data),
                    }
                ]
            },
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"Discord attachments error: {resp.status_code} {resp.text}")

        attachment = resp.json().get("attachments", [])[0]
        upload_url = attachment.get("upload_url")
        upload_filename = attachment.get("upload_filename")
        if not upload_url or not upload_filename:
            raise HTTPException(status_code=500, detail="Discord attachment response missing fields")

        try:
            put_response = httpx.put(
                upload_url,
                content=ogg_data,
                headers={"Content-Type": "audio/ogg"},
                timeout=30,
            )
        except RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upload to CDN failed: {e}")
        if put_response.status_code >= 400:
            raise HTTPException(status_code=put_response.status_code, detail=f"Upload URL error: {put_response.status_code} {put_response.text}")

        msg_response = httpx.post(
            f"{DISCORD_API}/channels/{target_channel}/messages",
            json={
                "flags": 8192,
                "attachments": [
                    {
                        "id": "0",
                        "filename": "voice-message.ogg",
                        "uploaded_filename": upload_filename,
                        "duration_secs": duration,
                        "waveform": waveform,
                    }
                ],
            },
            headers=HEADERS,
            timeout=30,
        )
        if msg_response.status_code >= 400:
            raise HTTPException(status_code=msg_response.status_code, detail=f"Send message error: {msg_response.status_code} {msg_response.text}")

        return {
            "status": "success",
            "duration_secs": duration,
            "file_size_bytes": len(ogg_data),
            "message": msg_response.json(),
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Discord API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/_debug_waveform")
async def debug_waveform(audio: UploadFile):
    """Debug endpoint: upload audio and get the generated waveform + raw values."""
    audio_data = await audio.read()
    ogg_data = convert_to_opus_ogg(audio_data)
    waveform_b64 = generate_waveform(ogg_data)
    
    # Also return raw values for inspection
    import base64
    raw_bytes = base64.b64decode(waveform_b64)
    raw_values = list(raw_bytes)
    
    return {
        "waveform_base64": waveform_b64,
        "raw_values": raw_values,
        "length": len(raw_values),
        "min": min(raw_values) if raw_values else 0,
        "max": max(raw_values) if raw_values else 0,
    }

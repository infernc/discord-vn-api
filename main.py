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
from scipy.signal import medfilt

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
    all_samples = []
    with av.open(io.BytesIO(audio_bytes)) as container:
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        for frame in container.decode(stream):
            all_samples.append(frame.to_ndarray().flatten().astype(np.float32))

    if not all_samples:
        return ""

    full_audio = np.abs(np.concatenate(all_samples))
    duration = len(full_audio) / sample_rate
    
    # 1. DYNAMIC BINNING
    num_bins = int(np.clip(duration * 14, 40, 256))
    raw_chunks = np.array_split(full_audio, num_bins)
    
    # Use Mean of the top 10% of each chunk (Robust Peak)
    heights = []
    for chunk in raw_chunks:
        if len(chunk) > 0:
            top_decile = np.sort(chunk)[-max(1, len(chunk)//10):]
            heights.append(np.mean(top_decile))
        else:
            heights.append(0)
    
    heights = np.array(heights)

    # 2. GLOBAL NORMALIZATION
    if np.max(heights) > 0:
        heights = heights / np.max(heights)

    # 3. DISCRETE QUANTIZATION (The Discord Secret)
    # We map the 0-1 range into 16 steps (0, 17, 34... 255)
    # This removes the 'erratic' tiny differences between bars.
    quantized = np.round(heights * 15) / 15
    
    # 4. NON-LINEAR LIFT (for the 'Short Bars')
    # We apply the lift to the quantized levels so mids stay strong.
    lifted = np.power(quantized, 0.5) * 255

    # 5. MEDIAN FILTERING
    # Unlike Gaussian, Median filtering keeps the 'pill' shape 
    # but removes single-bar spikes that look out of place.
    # kernel_size=3 means it looks at the bar before and after.
    smoothed = medfilt(lifted, kernel_size=3)

    # 6. FINAL FLOOR
    # Discord bars are never shorter than a specific 'dot' size.
    final_waveform = np.clip(smoothed, 30, 255).astype(np.uint8)

    return base64.b64encode(bytes(final_waveform)).decode('ascii')

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.webm', '.opus', '.mp4', '.wma'}

def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS

def is_audio_file(filename: str) -> bool:
    """Check if a file is audio based on its extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in AUDIO_EXTENSIONS

import traceback

@app.post("/voice-notes")
async def create_voice_note(audio: UploadFile = None, channel: str = Form(default="1")):
    """Create a voice note or post an image to a Discord channel."""
    try:
        # Select channel ID based on form input
        target_channel = CHANNEL_ID_1 if channel == "1" else (CHANNEL_ID_2 or CHANNEL_ID_1)
        
        if audio is None:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_data = await audio.read()
        if not file_data:
            raise HTTPException(status_code=400, detail="No file provided")

        # validate token early to provide clearer errors
        validate_bot_token()

        # Check if it's an image - post directly without conversion
        if is_image_file(audio.filename or ""):
            return await _post_image(target_channel, file_data, audio.filename or "image.png")
        
        # Otherwise treat as audio and convert to voice note
        return await _post_voice_note(target_channel, file_data)

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Discord API error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing error: {type(e).__name__}: {e}")


async def _post_image(target_channel: str, image_data: bytes, filename: str) -> dict:
    """Post an image directly to Discord."""
    # Determine content type
    ext = os.path.splitext(filename)[1].lower()
    content_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    content_type = content_type_map.get(ext, 'image/png')

    resp = httpx.post(
        f"{DISCORD_API}/channels/{target_channel}/attachments",
        json={
            "files": [
                {
                    "id": "0",
                    "filename": filename,
                    "file_size": len(image_data),
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
            content=image_data,
            headers={"Content-Type": content_type},
            timeout=30,
        )
    except RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upload to CDN failed: {e}")
    if put_response.status_code >= 400:
        raise HTTPException(status_code=put_response.status_code, detail=f"Upload URL error: {put_response.status_code} {put_response.text}")

    msg_response = httpx.post(
        f"{DISCORD_API}/channels/{target_channel}/messages",
        json={
            "attachments": [
                {
                    "id": "0",
                    "filename": filename,
                    "uploaded_filename": upload_filename,
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
        "file_size_bytes": len(image_data),
        "message": msg_response.json(),
    }


async def _post_voice_note(target_channel: str, audio_data: bytes) -> dict:
    """Convert audio to OGG/Opus and post as voice note to Discord."""
    ogg_data = convert_to_opus_ogg(audio_data)
    duration = get_duration(ogg_data)
    waveform = generate_waveform(ogg_data)

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

from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
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
    """Validate the Discord bot token by calling the /users/@me endpoint."""
    try:
        r = httpx.get(f"{DISCORD_API}/users/@me", headers=HEADERS, timeout=10)
    except RequestError as e:
        raise HTTPException(status_code=502, detail=f"Discord unreachable: {e}")
    if r.status_code == 401:
        raise HTTPException(status_code=401, detail="Discord authentication failed (401). Check BOT_TOKEN.")
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Discord /users/@me error: {r.status_code} {r.text}")
    return r.json()


def _decode_audio(audio_bytes: bytes) -> tuple[
    list[av.AudioFrame],  # original frames (for waveform)
    list[av.AudioFrame],  # resampled frames (for OGG/Opus)
    int,                  # duration in seconds (rounded up)
]:
    """Decode audio data once.

    Returns:
        * original frames – raw audio frames at the source sample rate.
        * resampled frames – frames converted to 48 kHz mono for Opus encoding.
        * duration – total length in whole seconds (minimum 1).
    """
    with av.open(io.BytesIO(audio_bytes)) as container:
        audio_stream = next(s for s in container.streams if s.type == "audio")

        # Duration from stream metadata if available
        if audio_stream.duration is not None and audio_stream.time_base is not None:
            duration = float(audio_stream.duration * audio_stream.time_base)
        else:
            duration = None

        # Prepare a resampler for the required Opus format
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=48_000,
        )

        orig_frames: list[av.AudioFrame] = []
        resampled_frames: list[av.AudioFrame] = []

        for frame in container.decode(audio=0):
            # Keep original frame for waveform
            orig_frames.append(frame)

            # Resample for OGG/Opus
            try:
                r = resampler.resample(frame)
                if r is None:
                    continue
                if isinstance(r, (list, tuple)):
                    resampled_frames.extend(r)
                else:
                    resampled_frames.append(r)
            except Exception:
                # If resampling fails, skip adding to resampled list but keep original
                pass

        # Fallback duration if header missing – use last original frame
        if duration is None and orig_frames:
            last = orig_frames[-1]
            duration = float(last.time * audio_stream.time_base)

        return orig_frames, resampled_frames, max(1, int(round(duration)))

def _rms_envelope_from_frames(frames: list[av.AudioFrame], step_ms: int = 10) -> np.ndarray:
    """Compute RMS envelope directly from decoded frames.

    The function processes frames sequentially, accumulating samples until a
    window of ``step_ms`` milliseconds is filled, then calculates the RMS of that
    window. This avoids building a full audio array in memory.
    """
    if not frames:
        return np.array([], dtype=np.float32)

    # Sample rate is taken from the first frame (all frames share the same rate)
    sample_rate = frames[0].sample_rate
    samples_per_step = int(sample_rate * (step_ms / 1000))
    if samples_per_step == 0:
        samples_per_step = 1

    envelope: list[float] = []
    buffer = np.empty(0, dtype=np.float32)

    for frame in frames:
        # Convert frame to mono float32 samples
        frame_arr = frame.to_ndarray().mean(axis=0).astype(np.float32)
        # Append to rolling buffer
        buffer = np.concatenate((buffer, frame_arr))
        # Extract full steps
        while buffer.size >= samples_per_step:
            segment = buffer[:samples_per_step]
            envelope.append(np.sqrt(np.mean(np.square(segment))))
            buffer = buffer[samples_per_step:]

    # Process any remaining samples as the final partial step
    if buffer.size:
        envelope.append(np.sqrt(np.mean(np.square(buffer))))

    return np.array(envelope, dtype=np.float32)


def convert_to_opus_ogg(frames: list[av.AudioFrame]) -> bytes:
    """Encode a list of audio frames into an OGG/Opus byte stream."""
    output_buf = io.BytesIO()
    with av.open(output_buf, "w", format="ogg") as out_file:
        out_stream = out_file.add_stream("libopus", rate=48_000)
        out_stream.layout = "mono"
        out_stream.bit_rate = 32_000

        pts_counter = 0
        for frame in frames:
            # Ensure pts is monotonic for the encoder
            frame.pts = pts_counter
            pts_counter += frame.samples
            for packet in out_stream.encode(frame):
                out_file.mux(packet)

        # Flush remaining packets
        for packet in out_stream.encode(None):
            out_file.mux(packet)

    output_buf.seek(0)
    return output_buf.read()



def generate_waveform(frames: list[av.AudioFrame]) -> str:
    """Generate a base64‑encoded SVG waveform from decoded audio frames.

    The RMS envelope is computed on‑the‑fly to avoid allocating a large audio
    buffer.
    """
    # Compute RMS envelope directly from frames
    envelope = _rms_envelope_from_frames(frames)

    if envelope.size == 0:
        return ""

    # Resample envelope to 256 bars for a fixed‑size waveform
    x_old = np.linspace(0, 1, envelope.size)
    x_new = np.linspace(0, 1, 256)
    rms_vals = np.interp(x_new, x_old, envelope)
    
    rms_vals /= (np.max(rms_vals) + 1e-9)
    a = 800
    waveform = (np.log(1 + a * rms_vals) / np.log(1 + a)) * 255
    
    # Limiter Knee
    threshold = 125
    ratio = 0.18
    limited = np.where(waveform > threshold, threshold + (waveform - threshold) * ratio, waveform)
    
    # Final Constraints
    final_val = np.clip(limited, 0, 185) # "Taller" request handled here
    final_val[rms_vals < 0.005] = 0
    
    # Gaussian smoothing
    final_waveform = np.convolve(final_val, [0.1, 0.8, 0.1], mode='same')

    return base64.b64encode(final_waveform.astype(np.uint8).tobytes()).decode('utf-8')

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.webm', '.opus', '.mp4', '.wma'}

def is_image_file(filename: str) -> bool:
    """Return ``True`` if the filename has a known image extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS

def is_audio_file(filename: str) -> bool:
    """Return ``True`` if the filename has a known audio extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in AUDIO_EXTENSIONS

import traceback

@app.post("/voice-notes")
async def create_voice_note(audio: UploadFile = None, channel: str = Form(default="1")):
    """Endpoint to receive an uploaded file and forward it to Discord.

    If the file is an image it is posted directly; otherwise it is processed as
    an audio voice note.
    """
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
    """Upload an image file to a Discord channel as an attachment."""
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
    """Convert raw audio to OGG/Opus, generate a waveform, and send as a Discord voice note."""
    # Single decode pass – get original frames, resampled frames, and duration
    orig_frames, resampled_frames, duration = _decode_audio(audio_data)

    # Encode the resampled frames to OGG/Opus
    ogg_data = convert_to_opus_ogg(resampled_frames)

    # Generate waveform from the original (non‑resampled) frames
    waveform = generate_waveform(orig_frames)

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
    """Simple health‑check endpoint returning ``{"status": "ok"}``."""
    return {"status": "ok"}


@app.post("/_debug_waveform")
async def debug_waveform(audio: UploadFile):
    """Debug endpoint that returns the generated waveform and raw byte values for an uploaded audio file."""
    audio_data = await audio.read()
    # Decode once – we only need the original frames for the waveform
    orig_frames, _, _ = _decode_audio(audio_data)
    waveform_b64 = generate_waveform(orig_frames)

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

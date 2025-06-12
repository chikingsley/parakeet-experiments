from pathlib import Path
from pydub import AudioSegment, silence
from nemo.collections.asr.models import EncDecMultiTaskModel
import torch, tempfile, math
from tqdm import tqdm

# ------------- user‑config -------------
MP3_PATH = Path("/Volumes/simons-enjoyment/GitHub/parakeet-mlx/French III - Lesson 01.mp3")
CHUNK_LEN_MS   = 30_000      # 30‑second windows
OVERLAP_MS     = 5_000       # 5‑second overlap
BEAM_SIZE      = 4           # small beam to reduce loops
# ---------------------------------------

def ms_to_timestamp(ms: int) -> str:
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1000
    ms_remainder = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_remainder:03d}"

with tempfile.TemporaryDirectory() as td:
    wav_path = Path(td) / "lesson01.wav"

    # --- convert MP3 → WAV (mono, 16 kHz) ---
    audio = AudioSegment.from_mp3(MP3_PATH).set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    print(f"Converted → {wav_path}")

    # --- load Canary on CPU, then move to MPS/FP16, beam search ---
    asr = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-flash", map_location="cpu")
    asr.to(device="mps", dtype=torch.float16)

    cfg = asr.cfg.decoding
    cfg.strategy = "beam"
    cfg.beam.beam_size = BEAM_SIZE
    asr.change_decoding_strategy(cfg)

    # --- chunk & transcribe ---
    step_ms = CHUNK_LEN_MS - OVERLAP_MS
    full_audio = AudioSegment.from_wav(wav_path)
    total_chunks = max(
        1,
        math.ceil((len(full_audio) - OVERLAP_MS) / (CHUNK_LEN_MS - OVERLAP_MS))
    )
    pbar = tqdm(total=total_chunks, desc="Transcribing", unit="chunk")
    srt_lines = []
    idx = 1

    for start_ms in range(0, len(full_audio), step_ms):
        pbar.update(1)
        chunk = full_audio[start_ms:start_ms + CHUNK_LEN_MS]
        if len(chunk) == 0:
            break

        # trim leading/trailing silence inside the chunk
        non_silent = silence.detect_nonsilent(chunk, min_silence_len=300, silence_thresh=-50)
        if non_silent:
            first_ns, last_ns = non_silent[0][0], non_silent[-1][1]
            chunk = chunk[first_ns:last_ns]

        chunk_path = Path(td) / f"chunk_{start_ms//1000}.wav"
        chunk.export(chunk_path, format="wav")

        text = asr.transcribe([str(chunk_path)], batch_size=1, return_hypotheses=False)[0].text.strip()
        if not text:
            continue

        start_ts = ms_to_timestamp(start_ms)
        end_ts   = ms_to_timestamp(min(start_ms + CHUNK_LEN_MS, len(full_audio)))

        srt_lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
        idx += 1

    pbar.close()

    print("\n--- SUBTITLE TRANSCRIPT ---\n")
    print("".join(srt_lines))
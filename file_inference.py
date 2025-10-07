import torch
from TTS.api import TTS
import os
import traceback

print("Starting inference...")

# --- 1. DEFINE YOUR PATHS ---
# Path to the directory where your model is saved
model_output_dir = "/home/chell/Documents/rebecca source/rebecca_IX/run_attempt1/GPT_XTTS_v2_Rebecca_FT_IX_Run1-September-12-2025_07+19PM-24b3b4d5/"

# The specific checkpoint .pth file you want to use
checkpoint_file = "checkpoint_41074.pth" # <<< Make sure this is the correct checkpoint name!

# Your reference audio file
speaker_wav_path = "/home/chell/Documents/rebecca source/rebecca_fine/rebecca_0016.wav" # <<< Make sure this is the correct absolute path!

# The text you want to convert to speech
text_to_speak = "This is a test of the model that I have finally managed to train."


# The name of the output audio file
output_filename = "NSA_test_output.wav"

# --- 2. CONSTRUCT FULL PATHS ---
#model_path = os.path.join(model_output_dir, checkpoint_file)
model_path = model_output_dir
config_path = os.path.join(model_output_dir, "config.json")

# --- 3. RUN THE MODEL ---
try:
    # --- File Checks ---
    if not os.path.exists(model_path): print(f"Error: Checkpoint file not found: {model_path}"); exit()
    if not os.path.exists(config_path): print(f"Error: Config file not found: {config_path}"); exit()
    if not os.path.exists(speaker_wav_path): print(f"Error: Speaker reference wav not found: {speaker_wav_path}"); exit()

    # --- Load TTS Model ---
    print("Loading TTS model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_path=model_path, config_path=config_path).to(device)
    print(f"Model loaded on {device}.")

    # --- Synthesize Speech ---
    print(f"Synthesizing: '{text_to_speak[:100]}...'") # Print first 100 chars
    tts.tts_to_file(
        text=text_to_speak,
        speaker_wav=speaker_wav_path,
        language="en",
        file_path=output_filename,
    )

    print(f"\nSUCCESS! Audio saved to: {os.path.abspath(output_filename)}")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    traceback.print_exc()
    print(f"---------------------------\n")

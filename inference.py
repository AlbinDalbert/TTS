import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths to your trained model
model_path = "/home/chell/Documents/rebecca source/rebecca_fine/run_attempt4_long/GPT_XTTS_v2_Rebecca_FT_Run2-May-03-2025_BEST_256G_140E_A1/best_model_32943.pth"
config_path = "/home/chell/Documents/rebecca source/rebecca_fine/run_attempt4_long/GPT_XTTS_v2_Rebecca_FT_Run2-May-03-2025_BEST_256G_140E_A1/config.json"
speaker_wav_path = "/home/chell/Documents/rebecca source/rebecca_fine/ref voice.wav"

# Text to synthesize
text_to_speak = "This is a test of my custom voice model, and hopefully, it will finally work."

# Output path for the generated audio
output_path = "my_test_speech.wav"

# --- Synthesize Speech ---
print("Loading model...")
# Init TTS with the model path and config path
tts = TTS(model_path=model_path, config_path=config_path).to(device)

print("Synthesizing speech...")
# Run TTS
tts.tts_to_file(
    text=text_to_speak,
    speaker_wav=speaker_wav_path,
    language="en", # Change this if your model is not English
    file_path=output_path
)

print(f"Speech synthesized and saved to {output_path}")

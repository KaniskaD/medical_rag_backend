import os
from faster_whisper import WhisperModel
from gtts import gTTS

ASR_MODEL_NAME = "tiny"
_whisper_model = None

LANG_CODES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te"
}

def get_asr_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(ASR_MODEL_NAME, device="cpu", compute_type="int8")
    return _whisper_model

def transcribe_audio_file(file_path: str, language_hint: str = "English") -> str:
    model = get_asr_model()
    target_lang_code = LANG_CODES.get(language_hint, "en")
    
    # task="transcribe" ensures output is in the native script
    segments, _ = model.transcribe(file_path, language=target_lang_code, task="transcribe")
    return " ".join([seg.text for seg in segments]).strip()

def text_to_speech(text: str, language_name: str, output_path: str):
    lang_code = LANG_CODES.get(language_name, "en")
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False
import requests
import json

# Ollama server + model name
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "phi3" 

def generate_text(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    """
    Calls local Ollama (Phi 3) with stop sequences to prevent echoing questions.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,  # Lower temperature is CRITICAL for script stability
            "top_p": 0.9,
            # Hard stop sequences to prevent the model from repeating input labels
            "stop": ["User:", "Question:", "Medical Agent:", "Answer:", "[USER]", "[SYSTEM]"]
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    except Exception as e:
        print("Ollama generation error:", repr(e))
        return "I encountered an error. Please try again."
import os

import requests


class OllamaGenerator:
    """Generador LLM usando Ollama (local)."""
    
    def __init__(self, model_name: str = "neural-chat", 
                 endpoint: str | None = None):
        self.model = model_name
        self.endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self.is_available = self._check_available()
    
    def _check_available(self) -> bool:
        """Verificar si Ollama está corriendo."""
        try:
            resp = requests.get(f"{self.endpoint}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
    
    def generate(self, prompt: str, temperature: float = 0.5, max_tokens: int = 512) -> str:
        """Generar respuesta usando Ollama."""
        if not self.is_available:
            return "⚠️ Ollama no está disponible. Ejecuta: ollama serve"
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"Error: {response.status_code}"
        
        except requests.Timeout:
            return "⚠️ Timeout: Ollama tardó demasiado en responder"
        except Exception as e:
            return f"Error generando respuesta: {e}"

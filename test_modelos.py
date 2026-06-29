import os
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar tu API KEY desde el .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ ERROR: No se encontró la GEMINI_API_KEY en el .env")
    exit()

genai.configure(api_key=API_KEY)

print("🔍 Interrogando a Google... Modelos disponibles para generar texto:\n")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ {m.name}")
except Exception as e:
    print(f"❌ Error al consultar: {e}")
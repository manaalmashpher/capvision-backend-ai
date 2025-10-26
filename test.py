from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


models = client.models.list()
print("Available models:", models)
response = client.models.generate_content(model='gemini-2.0-flash-lite', contents=["what is ai"])
print(response)
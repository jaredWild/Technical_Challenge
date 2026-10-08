from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key found:", api_key is not None)

client = genai.Client(api_key=api_key)

print("Sending request...")

# was initially using 3.8, there were spikes in demand 
#   that made it unusable
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Say hello world."
)

print("Response received.")
print(response.text)
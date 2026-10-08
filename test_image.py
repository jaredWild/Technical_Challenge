import cv2
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

prompt = """
Analyze this image as the initial observation for a visual memory system.

Return:
- entities: important visible people, animals, objects, or other notable things
- environment: a concise description of the general surroundings or setting
- change: "Initial observation."
- importance: a score from 1 to 10 representing how significant this observation would be to remember

Focus on information that would be useful to remember later.
Avoid unnecessary visual details.
"""

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

camera = cv2.VideoCapture(0)

success, frame = camera.read()
camera.release()

if not success:
    print("Failed to read from camera.")
    exit()

# encoding
success, buffer = cv2.imencode(".jpg", frame)

if not success:
    print("Failed to encode image.")
    exit()

image_bytes = buffer.tobytes()

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ),
        prompt
    ]
)



print(response.text)

cv2.imshow("Captured Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
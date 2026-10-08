import cv2
import time
import os
import threading
import queue
import json


from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from datetime import datetime


# i set the camera at 5 fps cause that's not too much 
# reduce down to quarter size
# I then also measure how much changed between frames
# only analyze ones that change more than a threshold

# ---------------------------------------------------PROMPTS----------------visual separation (this just helps me see things better)
# prompts

initial_prompt = """
Analyze this image as the initial observation for a visual memory system.

Identify:
- entities: important visible people, animals, objects, or other notable things
- environment: a concise description of the general surroundings or setting

Focus on information that would be useful to remember later.
Avoid unnecessary visual details.
"""
# old prompt
'''
initial_prompt = """
Analyze this image as the initial observation for a visual memory system.

Return:
- entities: important visible people, animals, objects, or other notable things
- environment: a concise description of the general surroundings or setting
- change: must be "Initial observation"
- importance: must be the integer 10 

Focus on information that would be useful to remember later.
Avoid unnecessary visual details.
"""
'''

change_prompt = """
Compare these two images as observations in a visual memory system.

The first image is the previous remembered observation.
The second image is the current observation.

Return:
- entities: important people, animals, objects, or other notable things visible in the CURRENT image
- environment: a concise description of the CURRENT surroundings or setting
- change: a concise description of the meaningful semantic difference between the previous and current observations
- importance: an integer from 1 to 10 representing how significant the change is:
  1-3 = minor or routine
  4-6 = meaningful
  7-8 = major or unusual
  9-10 = exceptional

Focus on meaningful changes in people, objects, or environment.
Do not treat small changes in framing, camera angle, lighting, or apparent distance as meaningful changes by themselves.
If the camera has moved into a genuinely different environment, that is a meaningful change.
"""

# -------------------------------------------------------SETTING UP--------------------------------------------------------------------------------

#time between sample in secs
SAMPLE_INTERVAL = 0.2  
last_sample_time = 0

RESIZE_SCALE = 0.25

# 5 was too high, 3.0 may still need to be lowered
CHANGE_THRESHOLD = 3.0

# compress quality
JPEG_QUALITY = 80

# prevent queue size to control ram usage
MAX_QUEUE_SIZE = 50

# for the return prompt
class MemoryAnalysis(BaseModel):
    entities: list[str]
    environment: str
    change: str
    importance: int

# api
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# for camera
previous_frame = None
initial_frame_queued = False


# frame queue
#  each item will be: (timestamp, encoded_jpeg_bytes)
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# ----------------------------------------------------IMAGES--------------------------------------------------------

# encoding frames to jpeg
def encode_frame(frame):

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not success:
        raise ValueError("Failed to encode image.")

    return buffer.tobytes()

# jpeg to image - for gemini
def image_part(image_bytes):

    return types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg"
    )

# ----------------------------------------ANALYSIS-------------------------------------------------------

def analyze_initial_frame(image_bytes):

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            image_part(image_bytes),
            initial_prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MemoryAnalysis
        )
    )

    memory = response.parsed

    # manually do it in case gemini goes weird
    memory.change = "Initial observation."
    memory.importance = 10

    return memory

# analyze changed between last analyzed image and this one
def analyze_change(previous_image_bytes, current_image_bytes):

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            change_prompt,

            "Previous observation:",
            image_part(previous_image_bytes),

            "Current observation:",
            image_part(current_image_bytes)
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MemoryAnalysis
        )
    )

    return response.parsed

# -----------------------------------------------------------threads-------------------------------------------------------

def analysis_worker():

    previous_analyzed_image = None

    while True:

        item = frame_queue.get()

        # none tells it to shut down
        if item is None:
            break

        timestamp, current_image = item

        try:

            if previous_analyzed_image is None:
                analysis = analyze_initial_frame(
                    current_image
                )

            else:
                analysis = analyze_change(
                    previous_analyzed_image,
                    current_image
                )

            # timestamp comes from comp, not gemini
            memory = {
                "timestamp": timestamp,
                **analysis.model_dump()
            }

            print("\n--- MEMORY CREATED ---")
            print(json.dumps(memory, indent=2))

            # Only update this if analysis succeeded
            previous_analyzed_image = current_image

        except Exception as e:
            print("\nAnalysis failed:", e)

        finally:
            frame_queue.task_done()


# start actual worker
worker = threading.Thread(
    target=analysis_worker,
    daemon=True
)

worker.start()

# -----------------------------------------------------------------------------------------------------------

camera = cv2.VideoCapture(0)


while True:
    # initial camera
    success, frame = camera.read()

    if not success:
        print("Failed to read from camera.")
        break

    cv2.imshow("Camera", frame)

    current_time = time.time()

    # sampling and resizing
    if current_time - last_sample_time >= SAMPLE_INTERVAL:

        resized_frame = cv2.resize(
            frame,
            None,
            fx=RESIZE_SCALE,
            fy=RESIZE_SCALE,
            interpolation=cv2.INTER_AREA
        )


        # ====================================================
        # INITIAL MEMORY
        # ====================================================
        
        # get the initial memory, and everything required with it

        if not initial_frame_queued:

            encoded_frame = encode_frame(frame)

            if encoded_frame is not None:

                timestamp = datetime.now().isoformat(
                    timespec="seconds"
                )

                frame_queue.put(
                    (timestamp, encoded_frame)
                )

                initial_frame_queued = True

                print("\nInitial frame queued.")


        # other frames
        elif previous_frame is not None:

            # check difference
            difference = cv2.absdiff(
                previous_frame,
                resized_frame
            )

            change_amount = difference.mean()

            # if change is big enough
            if change_amount >= CHANGE_THRESHOLD:

                encoded_frame = encode_frame(frame)

                if encoded_frame is not None:

                    timestamp = datetime.now().isoformat(
                        timespec="seconds"
                    )

                    # put on the queue if possible, wait until it is atm
                    #   will definitely change this later if I have time
                    try:
                        frame_queue.put_nowait(
                            (timestamp, encoded_frame)
                        )

                        print(
                            "Frame queued. Change:",
                            change_amount,
                            "Queue size:",
                            frame_queue.qsize()
                        )

                    except queue.Full:
                        print(
                            "Frame queue full - "
                            "dropping selected frame."
                        )

        # change to new frame and time
        previous_frame = resized_frame
        last_sample_time = current_time

    #exiting 
    key = cv2.waitKey(1)

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("\nCamera stopped.")
print("Processing remaining queued frames...")

# put an end marker for the worker 
frame_queue.put(None)

worker.join()

print("Finished.")

'''
def analyze_frame(current_frame, previous_analyzed_frame=None):

    current_image = frame_to_part(current_frame)

    if previous_analyzed_frame is None:
        contents = [
            current_image,
            initial_prompt
        ]

    else:
        previous_image = frame_to_part(previous_analyzed_frame)

        contents = [
            change_prompt,
            "Previous observation:",
            previous_image,
            "Current observation:",
            current_image
        ]

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Memory
        )
    )

    memory = response.parsed

    # manually enforce this cause it probably won't do it right, especially with the limited prompt
    if previous_analyzed_frame is None:
        memory.change = "Initial observation."
        memory.importance = 10

    return memory


executor = ThreadPoolExecutor(max_workers=1)

analysis_future = None
pending_analysis_frame = None

last_analyzed_frame = None
previous_frame = None

camera = cv2.VideoCapture(0)




while True: 
    success, frame = camera.read()

    # just for checking
    if not success:
        print("Failed to read from camera.")
        break

    
    cv2.imshow("Camera", frame)

    current_time = time.time()

    # getting frames
    if current_time - last_sample_time >= SAMPLE_INTERVAL:
        # print("Sampled frame")

        resized_frame = cv2.resize(
            frame,
            None,
            fx=RESIZE_SCALE,
            fy=RESIZE_SCALE,
            interpolation=cv2.INTER_AREA
        )

        

        # First memory
        if not initial_memory_created:
            print("\n--- INITIAL MEMORY ---")
            result = analyze_frame(frame, initial_prompt)

            if result is not None:
                print(result)
                print("change: Initial observation.")
                print("importance: 10")
                initial_memory_created = True
                last_analysis_time = current_time

        else:
            # Compare current sampled frame to previous sampled frame
            if previous_frame is not None:
                difference = cv2.absdiff(previous_frame, resized_frame)
                change_amount = difference.mean()

                if change_amount >= CHANGE_THRESHOLD:
                    print("\n--- CHANGE DETECTED ---")
                    print("change_amount:", change_amount)
                    
                    result = analyze_frame(frame, change_prompt)
                    
                    if result is not None:
                        print(result)
                        
                    last_analysis_time = current_time

        previous_frame = resized_frame

        last_sample_time = current_time

    key = cv2.waitKey(1)

    # quit with q as necessary
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
'''
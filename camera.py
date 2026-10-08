import cv2
import time
import os
import threading
import queue
import json
import sqlite3
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from datetime import datetime


# i set the camera at 5 fps cause that's not too much 
# reduce down to quarter size
# I then also measure how much changed between frames
# only analyze ones that change more than a threshold

# importance may need to be adjusted, but its hard to tell considering I was simply testing myself 
#   moving in a room or walking around with the camera

# ---------------------------------------------------PROMPTS----------------visual separation (this just helps me see things better)
# prompts

initial_prompt = """
Analyze this image as the initial observation for a visual memory system.

Identify:
- entities: important visible people, animals, objects, or other notable things
- environment: a concise description of the general surroundings or setting
- change: must be "Initial observation"
- importance: must be the integer 10 

Focus on information that would be useful to remember later.
Avoid unnecessary visual details.
"""

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

# i have no idea what threshold to put this at. in my room with just me moving around, 5-10 works fairly well. 
#   it would need to be changed depending on context
CHANGE_THRESHOLD = 20.0

# compress quality
JPEG_QUALITY = 80

# prevent queue size to control ram usage
MAX_QUEUE_SIZE = 50


# for subtle movements - timing
SLOW_CHANGE_INTERVAL = 10.0
last_stored_time = None
last_slow_check_time = time.time()

# last frame and limiting to only 1 thread
last_stored_frame = None
last_stored_frame_lock = threading.Lock()

# for the return prompt
class MemoryAnalysis(BaseModel):
    entities: list[str]
    environment: str
    change: str
    importance: int = Field(ge=1, le=10)

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


# -----------------------------------------------------------THREADS-------------------------------------------------------

def analysis_worker():
    previous_analyzed_image = None
    global last_stored_frame, last_stored_time

    # for memories
    connection = sqlite3.connect("memories.db")

    connection.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            entities TEXT,
            environment TEXT,
            change TEXT,
            importance INTEGER,
            image BLOB
        )
    """)

    connection.commit()


    # then on to standard work
    while True:

        item = frame_queue.get()

        # none tells it to shut down
        if item is None:
            frame_queue.task_done()
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

            # saving the memories
            connection.execute(
                """
                INSERT INTO memories
                (timestamp, entities, environment, change, importance, image)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory["timestamp"],
                    json.dumps(memory["entities"]),
                    memory["environment"],
                    memory["change"],
                    memory["importance"],
                    current_image
                )
            )

            connection.commit()

            # decode it (yeah, i know it's ineffecient, but that could be fixed later)
            decoded_image = cv2.imdecode(
                np.frombuffer(current_image, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            # resize
            stored_resized_frame = cv2.resize(
                decoded_image,
                None,
                fx=RESIZE_SCALE,
                fy=RESIZE_SCALE,
                interpolation=cv2.INTER_AREA
            )

            
            with last_stored_frame_lock:
                last_stored_frame = stored_resized_frame
                last_stored_time = time.time()


            print("\n--- MEMORY CREATED ---")
            print(json.dumps(memory, indent=2))

            # Only update this if analysis succeeded
            previous_analyzed_image = current_image

        except Exception as e:
            print("\nAnalysis failed:", e)

        finally:
            frame_queue.task_done()

    connection.close()


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

        # this is to prevent double-saves
        frame_queued = False

        # ====================================================
        # INITIAL MEMORY
        # ====================================================
        
        # get the initial memory when starting the camera, and everything required with it

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

                    # put on the queue if possible, don't 
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
                        frame_queued = True

                    except queue.Full:
                        print(
                            "Frame queue full - "
                            "dropping selected frame."
                        )
            # check if it's been a long time
            if (not frame_queued 
                and current_time - last_slow_check_time >= SLOW_CHANGE_INTERVAL
            ):

                
                # get last frame
                with last_stored_frame_lock:
                    stored_frame = (
                        last_stored_frame.copy()
                        if last_stored_frame is not None
                        else None
                    )

                    # get last time
                    stored_time = last_stored_time

                #check if enough time has passed
                if (stored_frame is not None
                    and stored_time is not None
                    and current_time - stored_time >= SLOW_CHANGE_INTERVAL
                ):
                    # get difference
                    slow_difference = cv2.absdiff(
                        stored_frame,
                        resized_frame
                    )

                    slow_change_amount = slow_difference.mean()

                    print(
                        "Slow change check:",
                        slow_change_amount
                    )

                    # save the image
                    if slow_change_amount >= CHANGE_THRESHOLD:
                        encoded_frame = encode_frame(frame)

                        timestamp = datetime.now().isoformat(
                            timespec="seconds"
                        )

                        try:
                            frame_queue.put_nowait(
                                (timestamp, encoded_frame)
                            )

                            print(
                                "Slow change detected. Frame queued. Change:",
                                slow_change_amount,
                                "Queue size:",
                                frame_queue.qsize()
                            )

                        except queue.Full:
                            print(
                                "Frame queue full - dropping selected frame."
                            )

                # Ddon't check till another interval has passed
                last_slow_check_time = current_time

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
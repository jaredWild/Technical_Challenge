import cv2
import time


# i get the camera at 5 fps cause that's not too much 
# reduce down to quarter size
# I then also measure how much changed between frames
# only analyze ones that change more than a threshold

# if I have time, I want to expand on it
# if no memory in x amount of time, check for diff since
#   last change cause something could be subtle

camera = cv2.VideoCapture(0)

#time between sample in secs
SAMPLE_INTERVAL = 0.2  
last_sample_time = 0

RESIZE_SCALE = 0.25

# 5 was too high, 3.0 may still need to be lowered
CHANGE_THRESHOLD = 3.0
previous_frame = None



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

        #checking for differences between frames
        if previous_frame is not None:
             difference = cv2.absdiff(previous_frame, resized_frame)
             change_amount = difference.mean()
             print(change_amount)

             if change_amount > CHANGE_THRESHOLD:
                 
                 print("Frame will be analyzed")

        previous_frame = resized_frame

        last_sample_time = current_time

    key = cv2.waitKey(1)

    # quit with q as necessary
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
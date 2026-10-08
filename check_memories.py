import sqlite3
import cv2
import numpy as np
import json

connection = sqlite3.connect("memories.db")

rows = connection.execute("""
    SELECT id, timestamp, entities, environment, change, importance, image
    FROM memories
    ORDER BY id
""").fetchall()

connection.close()

for row in rows:
    memory_id = row[0]
    timestamp = row[1]
    entities = json.loads(row[2])
    environment = row[3]
    change = row[4]
    importance = row[5]
    image_bytes = row[6]

    print("\n" + "=" * 60)
    print(f"Memory ID: {memory_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Entities: {entities}")
    print(f"Environment: {environment}")
    print(f"Change: {change}")
    print(f"Importance: {importance}")
    print("=" * 60)

    if image_bytes is not None:
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        cv2.imshow(f"Memory {memory_id}", image)

        print("Press any key for next memory, or Q to quit.")

        key = cv2.waitKey(0)

        cv2.destroyAllWindows()

        if key == ord("q"):
            break
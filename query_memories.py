import sqlite3
import json
import cv2
import numpy as np

# this one seems really self-explanatory to me

connection = sqlite3.connect("memories.db")

def search_memories(search_term):
    search = f"%{search_term}%"

    return connection.execute(
        """
        SELECT id, timestamp, entities, environment, change, importance
        FROM memories
        WHERE entities LIKE ?
           OR environment LIKE ?
           OR change LIKE ?
        ORDER BY timestamp DESC
        """,
        (search, search, search)
    ).fetchall()


def recent_memories(limit=5):
    return connection.execute(
        """
        SELECT id, timestamp, entities, environment, change, importance
        FROM memories
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()


def important_memories(min_importance=7):
    return connection.execute(
        """
        SELECT id, timestamp, entities, environment, change, importance
        FROM memories
        WHERE importance >= ?
        ORDER BY importance DESC, timestamp DESC
        """,
        (min_importance,)
    ).fetchall()


def last_seen(search_term):
    search = f"%{search_term}%"

    return connection.execute(
        """
        SELECT id, timestamp, entities, environment, change, importance
        FROM memories
        WHERE entities LIKE ?
           OR environment LIKE ?
           OR change LIKE ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (search, search, search)
    ).fetchone()


def memory_by_id(memory_id):
    return connection.execute(
        """
        SELECT id, timestamp, entities, environment, change, importance, image
        FROM memories
        WHERE id = ?
        """,
        (memory_id,)
    ).fetchone()


def print_memory(row):
    if row is None:
        print("No matching memory found.")
        return

    memory_id, timestamp, entities, environment, change, importance = row

    entities = json.loads(entities)

    print("\n" + "=" * 50)
    print("Memory ID:", memory_id)
    print("Timestamp:", timestamp)
    print("Entities:", entities)
    print("Environment:", environment)
    print("Change:", change)
    print("Importance:", importance)
    print("=" * 50)


def print_memories(rows):
    if not rows:
        print("No matching memories found.")
        return

    for row in rows:
        print_memory(row)

def view_memory(row):
    if row is None:
        print("No memory found with that ID.")
        return

    memory_id, timestamp, entities, environment, change, importance, image_bytes = row

    entities = json.loads(entities)

    print("\n" + "=" * 50)
    print("Memory ID:", memory_id)
    print("Timestamp:", timestamp)
    print("Entities:", entities)
    print("Environment:", environment)
    print("Change:", change)
    print("Importance:", importance)
    print("=" * 50)

    if image_bytes is not None:
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is not None:
            cv2.imshow(f"Memory {memory_id}", image)

            print("Press any key to close the image.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print("Image could not be decoded.")

    else:
        print("This memory has no stored image.")


# -------------------------------------------------------
# MENU
# -------------------------------------------------------

while True:

    print("\nWhat would you like to do?")
    print("1. Search memories by term")
    print("2. Show most recent memories")
    print("3. Show important memories")
    print("4. Find when something was last seen")
    print("5. View memory by ID")
    print("6. Exit")

    choice = input("\nChoose an option: ")

    # Search memories
    if choice == "1":
        search_term = input("Enter a search term: ")

        rows = search_memories(search_term)
        print_memories(rows)


    # Recent memories
    elif choice == "2":
        limit = input(
            "How many recent memories would you like to see? "
        )

        try:
            limit = int(limit)

            rows = recent_memories(limit)
            print_memories(rows)

        except ValueError:
            print("Please enter a number.")


    # Important memories
    elif choice == "3":
        min_importance = input(
            "Minimum importance (1-10): "
        )

        try:
            min_importance = int(min_importance)

            if 1 <= min_importance <= 10:
                rows = important_memories(min_importance)
                print_memories(rows)

            else:
                print(
                    "Importance must be between 1 and 10."
                )

        except ValueError:
            print("Please enter a number.")


    # Last seen
    elif choice == "4":
        search_term = input(
            "What would you like to find the last appearance of? "
        )

        row = last_seen(search_term)
        print_memory(row)


    # View specific memory with image
    elif choice == "5":
        memory_id = input(
            "Enter memory ID: "
        )

        try:
            memory_id = int(memory_id)

            row = memory_by_id(memory_id)
            view_memory(row)

        except ValueError:
            print("Please enter a valid numeric ID.")


    # Exit
    elif choice == "6":
        print("Exiting.")
        break


    else:
        print(
            "Invalid option. Please choose 1-6."
        )


connection.close()
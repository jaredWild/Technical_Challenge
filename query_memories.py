import sqlite3
import json

# this one all feels pretty self-explanatory to me

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


def print_memories(rows):
    if not rows:
        print("No matching memories found.")
        return

    for row in rows:
        print_memory(row)



while True:
    print("\nWhat would you like to do? Type the number desired.")
    print("1. Search memories by term")
    print("2. Show most recent memories")
    print("3. Show important memories")
    print("4. Find when something was last seen")
    print("5. Exit")

    choice = input("\nChoose an option: ")

    if choice == "1":
        search_term = input("Enter a search term: ")
        rows = search_memories(search_term)
        print_memories(rows)

    elif choice == "2":
        limit = input("How many recent memories would you like to see? ")

        try:
            limit = int(limit)
            rows = recent_memories(limit)
            print_memories(rows)
        except ValueError:
            print("Please enter a number.")

    elif choice == "3":
        min_importance = input(
            "Minimum importance (1-10): "
        )

        try:
            min_importance = int(min_importance)
            rows = important_memories(min_importance)
            print_memories(rows)
        except ValueError:
            print("Please enter a number.")

    elif choice == "4":
        search_term = input(
            "What would you like to find the last appearance of? "
        )

        row = last_seen(search_term)
        print_memory(row)

    elif choice == "5":
        print("Exiting.")
        break

    else:
        print("Invalid option. Please choose 1-5.")


connection.close()
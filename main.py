import os

# Paths (optional if your existing files already handle them)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_FOLDER = os.path.join(PROJECT_ROOT, "index")
os.makedirs(INDEX_FOLDER, exist_ok=True)

if __name__ == "__main__":
    print("1. Build Index")
    print("2. Start Chat")

    choice = input("Select option: ")

    if choice == "1":
        # Just run your existing build_index.py main code
        # If you want to call it directly, you could import and run the function
        # But if there's no single function, you can also run the script itself
        os.system("python src/build_index.py")
    elif choice == "2":
        # Run chat
        os.system("python src/chat.py")
    else:
        print("Invalid choice. Please select 1 or 2.")
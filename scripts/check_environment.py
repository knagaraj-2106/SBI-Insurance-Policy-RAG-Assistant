import os


def main():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY is not configured.")
        return

    print("✅ OPENAI_API_KEY is available.")
    print(f"Key detected: {api_key[:5]}********")


if __name__ == "__main__":
    main()
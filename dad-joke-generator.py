import requests
import json
import time

# Define the API endpoint
# API_URL = "http://host.docker.internal:11434/api/chat"
API_URL = "http://localhost:11434/api/chat"

# Set a retry mechanism for connection errors
MAX_RETRIES = 5
RETRY_DELAY = 5


def get_dad_joke(messages):
    """
    Sends a POST request to the API to get a dad joke,
    persisting the conversation history in the 'messages' list.

    Args:
        messages (list): A list of dictionaries representing the conversation history.

    Returns:
        list: The updated messages list including the new joke, or None if an error occurs.
    """
    payload = {
        "model": "my-dadjokes",
        "messages": messages,
        "options": {
            "temperature": 1.0
        }
    }

    print("Requesting a new dad joke...")
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Send the POST request with a timeout and stream the response
            response = requests.post(API_URL, json=payload, timeout=30, stream=True)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            full_response_content = ""
            # Iterate over the response lines to handle the stream of JSON objects
            for line in response.iter_lines():
                if line:
                    try:
                        # Decode the line and parse it as JSON
                        item = json.loads(line.decode('utf-8'))
                        # Assuming 'content' is the key for the generated text
                        full_response_content += item.get("message", {}).get("content", "")
                    except json.JSONDecodeError as e:
                        print(f"Skipping a malformed line: {e}")

            if full_response_content:
                print(f"Dad joke received: {full_response_content}")

                # Append the assistant's response to the message list for history persistence
                messages.append({
                    "role": "assistant",
                    "content": full_response_content
                })

                return messages
            else:
                print("Error: Empty or malformed response from the API.")
                return None

        except requests.exceptions.RequestException as e:
            retries += 1
            print(
                f"Error connecting to the API: {e}. Retrying in {RETRY_DELAY} seconds... (Attempt {retries}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)

    print("Failed to connect to the API after multiple retries. Please check if the service is running.")
    return None


if __name__ == "__main__":
    # The 'messages' list is the key to managing conversation history.
    messages = [
        {"role": "system",
         "content": "You are a creative dad joke generator. Do not repeat any jokes you have told before."},
        {"role": "user", "content": "Tell me a dad joke."}
    ]

    # Get the first joke
    updated_messages = get_dad_joke(messages)

    if updated_messages:
        messages = updated_messages

        # Add a new user prompt for the next joke request
        messages.append({"role": "user", "content": "Tell me another one."})

        # Get the second joke, which will use the updated message history
        updated_messages_2 = get_dad_joke(messages)

        if updated_messages_2:
            messages = updated_messages_2

            # You can repeat this process as many times as you want
            # to continue the conversation.
            print("\n--- Next Joke ---")
            messages.append({"role": "user", "content": "I'm ready for a third."})
            get_dad_joke(messages)

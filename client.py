import requests

def send_message_to_server(message):
    url = "http://127.0.0.1:5000/chat"
    try:
        response = requests.post(url, json={"message": message})
        data = response.json()
        print("Bot:", data.get("response", "No response from server."))
    except requests.exceptions.RequestException as e:
        print("Error communicating with server:", e)

if __name__ == '__main__':
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        send_message_to_server(user_input)
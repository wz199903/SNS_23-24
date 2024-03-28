import requests

def send_message_to_server(message):
    url = "http://127.0.0.1:5000/chat"
    try:
        response = requests.post(url, json={"message": message})
        data = response.json()
        print("Oracle:", data.get("response", "No response from server."))
    except requests.exceptions.RequestException as e:
        print("Error communicating with server:", e)

if __name__ == '__main__':
    print('Hello dear user, I am Oracle, a chatbot which can predict result of matches in UK Prime League in season 2023/2024, please provide me the team names and match date.')
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        send_message_to_server(user_input)
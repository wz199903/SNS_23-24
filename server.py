from flask import Flask, request, jsonify
app = Flask(__name__)

def get_model_response(message):
    predefined_answers = {
        "who is the champion": "A",
        "who is the winner": "B",
    }
    return predefined_answers.get(message.lower(), "Sorry, I don't understand thatW.")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    response = get_model_response(message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
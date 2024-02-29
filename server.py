from flask import Flask, request, jsonify
from difflib import SequenceMatcher
from dateutil import parser  
from datetime import datetime



app = Flask(__name__)


def check_similarity(a,b):
    return SequenceMatcher(None, a, b).ratio()


# an automatic date finder using NLP model
def get_date(message, mode):
    if mode == 0: 
        message_lower = message.lower()
        current_year = datetime.now().year
        try:
            date_return = str(parser.parse(message, fuzzy=True))
            return date_return
        except:
            if "this year" in message_lower:
                return current_year
            elif "last year" in message_lower:
                last_year = current_year - 1
                return last_year
            elif "next year" in message_lower:
                next_year = current_year + 1
                return next_year
    elif mode == 1:
        date_return = str(parser.parse(message, fuzzy=True))
        return date_return



def get_answer(index, date, result):
    if index == 0:
        return f"The champion of prime league of year {date} is {result}. How can I assist you further?"
    if index == 1:
        return f"The champion of match {date} is {result}. How can I assist you further?"



def get_model_response(message):


    predefined_keyword_set = {
        "champion of this last next year prime league": "A",
        "winner of match game beats": "B",
    }


    message_words = set(message.lower().split())
    best_match = None
    best_match_index = None
    highest_similarity = 0.0


    for index, question in enumerate(predefined_keyword_set.keys()):
        current_keyword_set  = question.split()
        similarity = sum(check_similarity(word, message_word) for word in current_keyword_set for message_word in message_words) / len(current_keyword_set)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = question
            best_match_index = index

    
    # date will be transfer to model

    date = get_date(message,best_match_index)
  


    if highest_similarity > 0.5:
        answer = get_answer(best_match_index, date, predefined_keyword_set[best_match])
        return answer
    else:
        return "Sorry, I don't understand that."



@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    response = get_model_response(message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify
from difflib import SequenceMatcher
from dateutil import parser  
from datetime import datetime
from predict import *
import subprocess

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
            else:
                return "error"
    elif mode == 1:
        date_return = str(parser.parse(message, fuzzy=True))
        return date_return



def get_answer(index, date, result):
    if date == "error":
        return f"Sorry, the year you are asking is invalid. How can I assist you further?"
    if index == 0:
        return f"The champion of prime league of {date} is {result}. How can I assist you further?"
    # if index == 1:
    #     return f"The champion of match {date} is {result}. How can I assist you further?"

# def get_result():
#     result = subprocess.run(['ls', '-l'], capture_output=True, text=True)
#     command = [
#     'python', 'predict.py',
#     '--predict_data', 'Datasets/test_set_en.csv',
#     '--load_model_path', 'path_to_save_model.pth',
#     '--use_cuda',
#     '--seq_len_required', '5'
#     ]

#     # Execute the command
#     result = subprocess.run(command, capture_output=True, text=True)

#     # Check if the command was executed successfully
#     if result.returncode == 0:
#         print("Command executed successfully!")
#         print("Output:\n", result.stdout)
#     else:
#         print("Error:", result.stderr)
    
#     return result

def extract_teams_from_sentence(sentence, team_names):
    found_teams = []
    for team in team_names:
        if team.lower() in sentence:
            found_teams.append(team)
        if len(found_teams) == 2:  # Assuming we only need to find two teams
            break
    return found_teams

def get_model_response(message):


    predefined_keyword_set = [
        # "champion of this last next year prime league": "A",
        "winner of match game beats"]
    

    Teams = ['Arsenal',
            'Aston Villa',
            'Bournemouth',
            'Brentford',
            'Brighton',
            'Burnley',
            'Chelsea',
            'Crystal Palace',
            'Everton',
            'Fulham',
            'Liverpool',
            'Luton',
            'Man City',
            'Man Utd',
            'Newcastle',
            'Nottingham Forest',
            'Sheffield Utd',
            'Spurs',
            'West Ham',
            'Wolves']

    



    message_words = set(message.lower().split())
    best_match = None
    best_match_index = None
    highest_similarity = 0.0
    found_teams = []

    for index, question in enumerate(predefined_keyword_set):
        current_keyword_set  = question.split()
        similarity = sum(check_similarity(word, message_word) for word in current_keyword_set for message_word in message_words) / len(current_keyword_set)
        if similarity > highest_similarity:
            print("in loop")
            highest_similarity = similarity
            best_match = question
            best_match_index = index
            found_teams = extract_teams_from_sentence(message_words, Teams)


    
    # date will be transfer to model

    date = get_date(message,best_match_index)
  
    seq_len = 5

    if highest_similarity > 0.5:
        
        # TODO Here
        result = predict_server(found_teams[0], found_teams[1], seq_len)
        print(found_teams)
        win = None
        if result[0] == 0:
            win = found_teams[0]
        elif result[0] == 2:
            win = found_teams[1]
        else:
            win = 'draw'
        

        answer = get_answer(result[0], date, win)
        return answer
    else:
        return "Sorry, I don't understand that, you can ask it this way: Who is winner between Man city and Aston Villa."



@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    response = get_model_response(message)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
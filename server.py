from flask import Flask, request, jsonify
from difflib import SequenceMatcher
from dateutil import parser  
from datetime import datetime
from predict import *


app = Flask(__name__)


def check_similarity(a,b):
    return SequenceMatcher(None, a, b).ratio()


def get_answer(index, date, result):
    date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    new_date = f"{date_obj.year}/{date_obj.month}/{date_obj.day}"
    return f"The winner on {new_date} is {result}. How can I assist you further?"


def extract_teams_from_sentence(sentence, team_names):
    normalized_sentence = sentence.lower()
    mentioned_teams = [team for team in team_names if team.lower() in normalized_sentence]
    return mentioned_teams

def get_model_response(message):


    predefined_keyword_set = [
        "winner of match game beats Premiere League"]
    

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
            'Nott\'m Forest',
            'Sheffield Utd',
            'Spurs',
            'West Ham',
            'Wolves',
            'Tottenham']

    



    message_words = set(message.lower().split())
    highest_similarity = 0.0
    found_teams = []

    for index, question in enumerate(predefined_keyword_set):
        current_keyword_set  = question.split()
        similarity = sum(check_similarity(word, message_word) for word in current_keyword_set for message_word in message_words) / len(current_keyword_set)
        if similarity > highest_similarity:
            highest_similarity = similarity
            found_teams = extract_teams_from_sentence(message, Teams)
            try:
                date = str(parser.parse(message, fuzzy=True))
            except:
                return "Sorry, Oracle cannot found a valid match date in your sentence, please try again"

    if len(found_teams) < 2:
        return "Sorry, please provide two valid team name."

    

    seq_len = 500

    if highest_similarity > 0.5:
        
        result = predict_server(date, seq_len, found_teams)
        try:
            result = result.cpu().tolist()
        except:
            if result == "ERROR":
                return "Sorry, the match info you have provided is invalid"

        win = None

        if result[0] == 2:
            win = "Home Team"
        elif result[0] == 0:
            win = "Away Team"
        else:
            win = 'draw'

        answer = get_answer(result[0], date, win)
        print(answer)
        return answer
    else:
        return "Sorry, I don't understand that, you can ask it this way: who wins the matches on 2024/2/1, between West Ham and Bournemouth ?"



@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    response = get_model_response(message)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
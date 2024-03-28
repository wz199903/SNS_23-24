# SNS_23-24 CW
## By SN: 21185548/ 23209239

### Overview
This project, **SNS Assignment Project 23-24**, implements a Oracle chatbot which predict the outcomes of UK Premier League matches from February 1st, 2023, to March 17th, 2023. The model uses LSTM (Long Short-Term Memory) neural network model for analysis

# Key Libraries
- **Python**: Version 3.9.x
- **Flask API**: For server and client implementation.
- **PyTorch**: For deep learning implementation.

### How to Run
1. Install the following setups. [Python Installation Guide](https://www.python.org/downloads/), [Flask Installation](https://flask.palletsprojects.com/en/2.0.x/installation/), and [PyTorch Installation](https://pytorch.org/get-started/locally/).
2. To start the Oracle Chatbot, please run `server.py` followed by the `client.py` in separate terminals.

### Data
Training and testing data are stored in **Datasets** repository:
  - `test_set_en.csv`
  - `train_set_en.csv`

### Training the model

The training process is in Jupyter Notebook: `train.ipynb`.

### Making Predictions

The prediction process, which will be called by server, is in `predict.py`.

### Oracle chatbot guide:

To properly give Oracle instruction, you have to input the correct match information like this:
 - who wins the matches on 2024/2/1, between West Ham and Bournemouth?


#### Teams List
Here is the team name list which Oracle can properly identiy:
- Arsenal
- Aston Villa
- Bournemouth
- Brentford
- Brighton
- Burnley
- Chelsea
- Crystal Palace
- Everton
- Fulham
- Liverpool
- Luton
- Man City
- Man Utd
- Newcastle
- Nott'm Forest
- Sheffield Utd
- Spurs
- Tottenham
- West Ham
- Wolves


#### Match Predictions Time Frame
The Oracle Chatbot can only predict matches from February 1, 2024, to March 17, 2024. 

- 2024/2/1	West Ham	Bournemouth
- 2024/2/1	Wolves	Man United
- 2024/2/3	Sheffield United	Aston Villa
- 2024/2/3	Newcastle	Luton
- 2024/2/3	Burnley	Fulham
- 2024/2/3	Brighton	Crystal Palace
- 2024/2/3	Everton	Tottenham
- 2024/2/4	Bournemouth	Nott'm Forest
- 2024/2/4	Chelsea	Wolves
- 2024/2/4	Man United	West Ham
- 2024/2/4	Arsenal	Liverpool
- 2024/2/5	Brentford	Man City
- 2024/2/10	Nott'm Forest	Newcastle
- 2024/2/10	Wolves	Brentford
- 2024/2/10	Tottenham	Brighton
- 2024/2/10	Man City	Everton
- 2024/2/10	Liverpool	Burnley
- 2024/2/10	Fulham	Bournemouth
- 2024/2/10	Luton	Sheffield United
- 2024/2/11	West Ham	Arsenal
- 2024/2/11	Aston Villa	Man United
- 2024/2/12	Crystal Palace	Chelsea
- 2024/2/17	Man City	Chelsea
- 2024/2/17	Brentford	Liverpool
- 2024/2/17	Burnley	Arsenal
- 2024/2/17	Fulham	Aston Villa
- 2024/2/17	Newcastle	Bournemouth
- 2024/2/17	Nott'm Forest	West Ham
- 2024/2/17	Tottenham	Wolves
- 2024/2/18	Luton	Man United
- 2024/2/18	Sheffield United	Brighton
- 2024/2/19	Everton	Crystal Palace
- 2024/2/20	Man City	Brentford
- 2024/2/21	Liverpool	Luton
- 2024/2/24	Aston Villa	Nott'm Forest
- 2024/2/24	Brighton	Everton
- 2024/2/24	Crystal Palace	Burnley
- 2024/2/24	Man United	Fulham
- 2024/2/24	Bournemouth	Man City
- 2024/2/24	Arsenal	Newcastle
- 2024/2/25	Wolves	Sheffield United
- 2024/2/26	West Ham	Brentford
- 2024/3/2	Luton	Aston Villa
- 2024/3/2	Nott'm Forest	Liverpool
- 2024/3/2	Newcastle	Wolves
- 2024/3/2	Tottenham	Crystal Palace
- 2024/3/2	Everton	West Ham
- 2024/3/2	Brentford	Chelsea
- 2024/3/2	Fulham	Brighton
- 2024/3/3	Burnley	Bournemouth
- 2024/3/3	Man City	Man United
- 2024/3/4	Sheffield United	Arsenal
- 2024/3/9	Arsenal	Brentford
- 2024/3/9	Wolves	Fulham
- 2024/3/9	Crystal Palace	Luton
- 2024/3/9	Man United	Everton
- 2024/3/9	Bournemouth	Sheffield United
- 2024/3/10	Aston Villa	Tottenham
- 2024/3/10	Brighton	Nott'm Forest
- 2024/3/10	West Ham	Burnley
- 2024/3/10	Liverpool	Man City
- 2024/3/11	Chelsea	Newcastle
- 2024/3/13	Bournemouth	Luton
- 2024/3/16	Luton	Nott'm Forest
- 2024/3/16	Fulham	Tottenham
- 2024/3/16	Burnley	Brentford
- 2024/3/17	West Ham	Aston Villa

import pandas as pd
import numpy as np


def load_and_process_season(season_suffix, standings):
    """
    Load a season's data, process it, and return the enhanced DataFrame.
    """
    data_path = f'./Datasets/season-{season_suffix}_1.csv'
    playing_stat = pd.read_csv(data_path)
    if 'Time' in playing_stat.columns:
        playing_stat.drop('Time', axis=1, inplace=True)
    playing_stat.drop('Referee', axis=1, inplace=True)

    all_columns = playing_stat.columns.tolist()
    start_index = all_columns.index('Date')
    end_index = all_columns.index('AC') + 1
    playing_stat = playing_stat.iloc[:, start_index:end_index]
    # Apply necessary transformations
    playing_stat['Date'] = pd.to_datetime(playing_stat['Date'])
    playing_stat.sort_values(by='Date', inplace=True)

    # Apply processing functions
    playing_stat = get_mw(playing_stat)
    playing_stat = get_gss(playing_stat)
    playing_stat = get_agg_points(playing_stat)
    playing_stat = wdl(playing_stat)
    playing_stat = add_form_df(playing_stat)
    playing_stat = get_last(playing_stat, standings)

    return playing_stat


def cumulative_goals_scored(playing_stat):
    goals_scored = {}
    for index, row in playing_stat.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals, away_goals = row['FTHG'], row['FTAG']

        if home_team not in goals_scored:
            goals_scored[home_team] = [0] * len(playing_stat)
        if away_team not in goals_scored:
            goals_scored[away_team] = [0] * len(playing_stat)

        goals_scored[home_team][index] = home_goals
        goals_scored[away_team][index] = away_goals

    for team in goals_scored:
        goals_scored[team] = pd.Series(goals_scored[team]).cumsum()

    df_goals_scored = pd.DataFrame(goals_scored)
    return df_goals_scored

def cumulative_goals_conceded(playing_stat):
    goals_conceded = {}
    for index, row in playing_stat.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals_conceded, away_goals_conceded = row['FTAG'], row['FTHG']

        if home_team not in goals_conceded:
            goals_conceded[home_team] = [0] * len(playing_stat)
        if away_team not in goals_conceded:
            goals_conceded[away_team] = [0] * len(playing_stat)

        goals_conceded[home_team][index] = home_goals_conceded
        goals_conceded[away_team][index] = away_goals_conceded

    for team in goals_conceded:
        goals_conceded[team] = pd.Series(goals_conceded[team]).cumsum()

    df_goals_conceded = pd.DataFrame(goals_conceded)
    return df_goals_conceded

def get_gss(playing_stat):
    GS = cumulative_goals_scored(playing_stat)
    GC = cumulative_goals_conceded(playing_stat)

    HTGS = [GS.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(playing_stat) + 1), playing_stat['HomeTeam'])]
    ATGS = [GS.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(playing_stat) + 1), playing_stat['AwayTeam'])]
    HTGC = [GC.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(playing_stat) + 1), playing_stat['HomeTeam'])]
    ATGC = [GC.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(playing_stat) + 1), playing_stat['AwayTeam'])]

    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC

    return playing_stat


def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    elif result == 'L':
        return 0


def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat


def get_matchres(playing_stat):
    matchres = []
    for index, row in playing_stat.iterrows():
        home_goals, away_goals = row['FTHG'], row['FTAG']
        if home_goals > away_goals:
            matchres.append({'Team': row['HomeTeam'], 'Result': 'W', 'Matchweek': row['MW']})
            matchres.append({'Team': row['AwayTeam'], 'Result': 'L', 'Matchweek': row['MW']})
        elif home_goals < away_goals:
            matchres.append({'Team': row['HomeTeam'], 'Result': 'L', 'Matchweek': row['MW']})
            matchres.append({'Team': row['AwayTeam'], 'Result': 'W', 'Matchweek': row['MW']})
        else:
            matchres.append({'Team': row['HomeTeam'], 'Result': 'D', 'Matchweek': row['MW']})
            matchres.append({'Team': row['AwayTeam'], 'Result': 'D', 'Matchweek': row['MW']})
    return pd.DataFrame(matchres)

# Function to calculate cumulative points from match results
def get_cuml_points(matchres):
    matchres['Points'] = matchres['Result'].apply(get_points)
    cuml_points = (matchres.pivot_table(index='Matchweek', columns='Team', values='Points', aggfunc='sum')
                   .fillna(0).cumsum())
    return cuml_points.astype(int)

# Function to assign cumulative points to playing_stat, excluding current match
def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cuml_points = get_cuml_points(matchres)
    HTP, ATP = [], []
    for index, row in playing_stat.iterrows():
        current_mw = row['MW']
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        if current_mw > 1:
            prev_mw = current_mw - 1
            HTP.append(cuml_points.loc[prev_mw, home_team] if prev_mw in cuml_points.index and home_team in cuml_points.columns else 0)
            ATP.append(cuml_points.loc[prev_mw, away_team] if prev_mw in cuml_points.index and away_team in cuml_points.columns else 0)
        else:
            HTP.append(0)
            ATP.append(0)
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat


def wdl(playing_stat):
    # Initialize columns for cumulative wins, draws, and losses for home and away teams
    playing_stat['HTW'] = 0
    playing_stat['ATW'] = 0
    playing_stat['HTD'] = 0
    playing_stat['ATD'] = 0
    playing_stat['HTL'] = 0
    playing_stat['ATL'] = 0

    # Track cumulative statistics for each team
    team_stats = {team: {'W': 0, 'D': 0, 'L': 0} for team in pd.concat([playing_stat['HomeTeam'], playing_stat['AwayTeam']]).unique()}

    for index, row in playing_stat.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        result = row['FTR']

        # Update the playing_stat DataFrame with cumulative stats before updating them
        playing_stat.at[index, 'HTW'] = team_stats[home_team]['W']
        playing_stat.at[index, 'ATW'] = team_stats[away_team]['W']
        playing_stat.at[index, 'HTD'] = team_stats[home_team]['D']
        playing_stat.at[index, 'ATD'] = team_stats[away_team]['D']
        playing_stat.at[index, 'HTL'] = team_stats[home_team]['L']
        playing_stat.at[index, 'ATL'] = team_stats[away_team]['L']

        # Update team_stats based on the match result
        if result == 'H':
            team_stats[home_team]['W'] += 1
            team_stats[away_team]['L'] += 1
        elif result == 'D':
            team_stats[home_team]['D'] += 1
            team_stats[away_team]['D'] += 1
        elif result == 'A':
            team_stats[home_team]['L'] += 1
            team_stats[away_team]['W'] += 1
    for team in team_stats:
        final_index = playing_stat[(playing_stat['HomeTeam'] == team) | (playing_stat['AwayTeam'] == team)].index[-1]
        playing_stat.at[final_index, 'HTW' if playing_stat.at[final_index, 'HomeTeam'] == team else 'ATW'] = \
            team_stats[team]['W']
        playing_stat.at[final_index, 'HTD' if playing_stat.at[final_index, 'HomeTeam'] == team else 'ATD'] = \
            team_stats[team]['D']
        playing_stat.at[final_index, 'HTL' if playing_stat.at[final_index, 'HomeTeam'] == team else 'ATL'] = \
            team_stats[team]['L']

    return playing_stat


def get_form(playing_stat, num):
    form_final = pd.DataFrame()
    matchres = get_matchres(playing_stat)  # Use your provided function
    teams = pd.concat([playing_stat['HomeTeam'], playing_stat['AwayTeam']]).unique()

    for team in teams:
        team_matches = matchres[matchres['Team'] == team].sort_values(by='Matchweek')
        results = team_matches['Result'].tolist()

        form_list = []
        for i in range(1, len(results) + 1):
            form = ''.join(results[max(0, i - num):i])
            form = form.rjust(num, 'M')  # Ensure left padding with 'M'
            form_list.append(form)

        team_form_df = pd.DataFrame({
            'Team': team,
            'Matchweek': team_matches['Matchweek'],
            'Form': form_list
        })

        form_final = pd.concat([form_final, team_form_df], ignore_index=True)

    return form_final.sort_values(by=['Team', 'Matchweek'])


def add_form(playing_stat, num):
    form_df = get_form(playing_stat, num)
    h, a = [], []

    for index, row in playing_stat.iterrows():
        matchweek = row['MW']
        home_team, away_team = row['HomeTeam'], row['AwayTeam']

        home_form_entry = form_df[(form_df['Team'] == home_team) & (form_df['Matchweek'] == matchweek - 1)]
        away_form_entry = form_df[(form_df['Team'] == away_team) & (form_df['Matchweek'] == matchweek - 1)]

        home_form = ['M'] * num
        away_form = ['M'] * num

        if not home_form_entry.empty:
            home_form = list(home_form_entry['Form'].values[0])[:num]
        if not away_form_entry.empty:
            away_form = list(away_form_entry['Form'].values[0])[:num]

        home_form_dict = {f'HM{i}': result for i, result in enumerate(home_form, 1)}
        away_form_dict = {f'AM{i}': result for i, result in enumerate(away_form, 1)}

        h.append(home_form_dict)
        a.append(away_form_dict)

    home_form_df = pd.DataFrame(h)
    away_form_df = pd.DataFrame(a)

    playing_stat = pd.concat([playing_stat, home_form_df, away_form_df], axis=1)

    return playing_stat


def add_form_df(playing_statistics, num_weeks=5):
    for i in range(1, num_weeks + 1):
        playing_statistics[f'HM{i}'] = None
        playing_statistics[f'AM{i}'] = None

    playing_statistics.sort_values(by='MW', inplace=True)
    team_form = {team: ['M'] * num_weeks for team in
                 set(playing_statistics['HomeTeam']) | set(playing_statistics['AwayTeam'])}

    for index, row in playing_statistics.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']

        home_result = 'W' if home_goals > away_goals else ('L' if home_goals < away_goals else 'D')
        away_result = 'L' if home_goals > away_goals else ('W' if home_goals < away_goals else 'D')

        # Insert the current form at the beginning for each team's list and then remove the last element to maintain the size
        for i in range(1, num_weeks + 1):
            playing_statistics.at[index, f'HM{i}'] = team_form[home_team][-i]  # Past form without including current match
            playing_statistics.at[index, f'AM{i}'] = team_form[away_team][-i]  # Past form without including current match

        team_form[home_team].append(home_result)  # Update form with current match result
        team_form[away_team].append(away_result)  # Update form with current match result

        team_form[home_team] = team_form[home_team][-num_weeks:]  # Keep only the last num_weeks entries
        team_form[away_team] = team_form[away_team][-num_weeks:]  # Keep only the last num_weeks entries

    return playing_statistics

def get_last(playing_stat, standings):
    # Adjust function to use the match date minus one year for the ranking
    playing_stat['Date'] = pd.to_datetime(playing_stat['Date'])
    playing_stat['Season_End_Year'] = playing_stat['Date'].apply(lambda x: x.year if x.month > 5 else x.year-1)

    team_to_lp = {}
    for year in playing_stat['Season_End_Year'].unique():
        previous_season_standings = standings[standings['Season_End_Year'] == year]
        team_to_lp.update(previous_season_standings.set_index('Team')['Rk'].to_dict())

    playing_stat['HomeTeamRk'] = playing_stat.apply(lambda x: team_to_lp.get(x['HomeTeam'], 20), axis=1)
    playing_stat['AwayTeamRk'] = playing_stat.apply(lambda x: team_to_lp.get(x['AwayTeam'], 20), axis=1)

    playing_stat['HomeTeamRk'] = playing_stat['HomeTeamRk'].astype(int)
    playing_stat['AwayTeamRk'] = playing_stat['AwayTeamRk'].astype(int)

    return playing_stat


def form_and_streaks(seasons_data):
    # Concatenate all seasons data and reset index to ensure unique indices
    playing_stat = pd.concat(seasons_data.values()).reset_index(drop=True)

    # Fill missing match form records with 'M' for the initial rounds
    for i in range(1, 6):
        playing_stat[f'HM{i}'] = playing_stat[f'HM{i}'].fillna('M')
        playing_stat[f'AM{i}'] = playing_stat[f'AM{i}'].fillna('M')

    # Concatenate the form strings for both home and away teams
    playing_stat['HTFormPtsStr'] = playing_stat[['HM1', 'HM2', 'HM3', 'HM4', 'HM5']].agg(''.join, axis=1)
    playing_stat['ATFormPtsStr'] = playing_stat[['AM1', 'AM2', 'AM3', 'AM4', 'AM5']].agg(''.join, axis=1)

    # Calculate form points
    def get_form_points(form_str):
        points = {'W': 3, 'D': 1, 'L': 0, 'M': 0}  # 'M' for matches not played yet
        return sum(points[result] for result in form_str if result in points)

    playing_stat['HomeFormPoints'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
    playing_stat['AwayFormPoints'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

    # Identify streaks
    def get_3game_ws(form_str):
        return 'WWW' in form_str

    def get_5game_ws(form_str):
        return 'WWWWW' in form_str

    def get_3game_ls(form_str):
        return 'LLL' in form_str

    def get_5game_ls(form_str):
        return 'LLLLL' in form_str

    playing_stat['Home3GameWinStreak'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws).astype(int)
    playing_stat['Away3GameWinStreak'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws).astype(int)
    playing_stat['Home5GameWinStreak'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws).astype(int)
    playing_stat['Away5GameWinStreak'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws).astype(int)
    playing_stat['Home3GameLossStreak'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls).astype(int)
    playing_stat['Away3GameLossStreak'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls).astype(int)
    playing_stat['Home5GameLossStreak'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls).astype(int)
    playing_stat['Away5GameLossStreak'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls).astype(int)

    return playing_stat


def stat_difference(playing_stat):
    # Goal difference
    playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
    playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

    # Points differences
    playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
    playing_stat['DiffFormPts'] = playing_stat['HomeFormPoints'] - playing_stat['AwayFormPoints']

    # Difference in last year ranks
    playing_stat['DiffLP'] = playing_stat['HomeTeamRk'] - playing_stat['AwayTeamRk']

    return playing_stat


def normalise_data(playing_stat):
    # Normalize goal statistics columns by their maximum values
    for col in ['HTGS', 'ATGS', 'HTGC', 'ATGC']:
        max_value = playing_stat[col].max()
        if max_value != 0:  # Avoid division by zero
            playing_stat[col + '_norm'] = playing_stat[col] / max_value

    # Normalize goal and points differences by the highest observed absolute value for the matchweek
    for col in ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']:
        # Create a new column for normalized values
        playing_stat[col] = playing_stat[col].astype(float)
        playing_stat[col + '_norm'] = 0.0

        # Loop through each matchweek to normalise based on matchweek data
        for mw in playing_stat['MW'].unique():
            mw_mask = playing_stat['MW'] == mw
            max_abs_diff = playing_stat.loc[mw_mask, col].abs().max()
            if max_abs_diff != 0:  # Avoid division by zero
                playing_stat.loc[mw_mask, col + '_norm'] = playing_stat.loc[mw_mask, col] / max_abs_diff

    # Normalize cumulative points by the maximum possible points so far
    playing_stat['HTP_norm'] = playing_stat.apply(lambda x: x['HTP'] / (x['MW'] * 3), axis=1)
    playing_stat['ATP_norm'] = playing_stat.apply(lambda x: x['ATP'] / (x['MW'] * 3), axis=1)

    return playing_stat


# Load the dataset
league_table_path = './Datasets/premier-league-tables.csv'
standings = pd.read_csv(league_table_path)
season_suffixes = ['0910', '1011', '1112', '1213', '1314', '1415', '1516',
                   '1617', '1718', '1819', '1920', '2021', '2122', '2223']
seasons_data = {}
for suffix in season_suffixes:
    season_key = f'20{suffix[:2]}-20{suffix[2:]}'
    seasons_data[season_key] = load_and_process_season(suffix, standings)

season_2009_2010_data = seasons_data['2009-2010']
season_2010_2011_data = seasons_data['2010-2011']
season_2011_2012_data = seasons_data['2011-2012']
season_2012_2013_data = seasons_data['2012-2013']
season_2013_2014_data = seasons_data['2013-2014']
season_2014_2015_data = seasons_data['2014-2015']
season_2015_2016_data = seasons_data['2015-2016']
season_2016_2017_data = seasons_data['2016-2017']
season_2017_2018_data = seasons_data['2017-2018']
season_2018_2019_data = seasons_data['2018-2019']
season_2019_2020_data = seasons_data['2019-2020']
season_2020_2021_data = seasons_data['2020-2021']
season_2021_2022_data = seasons_data['2021-2022']
season_2022_2023_data = seasons_data['2022-2023']

playing_stat = form_and_streaks(seasons_data)
playing_stat = stat_difference(playing_stat)
playing_stat = normalise_data(playing_stat)
playing_stat_train = playing_stat[:3800]
playing_stat_test = playing_stat[3800:]

playing_stat_train.to_csv('./Datasets/train_set.csv', index=False)
playing_stat_test.to_csv('./Datasets/test_set.csv', index=False)


import pandas as pd


def load_and_process_season(season_suffix, standings):
    """
    Load a season's data, process it, and return the DataFrame.
    :param season_suffix: suffix of the season
    :param standings: standings of Premier League
    :return match_data: dataframe with processed features added
    """
    data_path = f'./Datasets/season-{season_suffix}_1.csv'
    match_data = pd.read_csv(data_path)
    if 'Time' in match_data.columns:
        match_data.drop('Time', axis=1, inplace=True)
    match_data.drop('Referee', axis=1, inplace=True)

    all_columns = match_data.columns.tolist()
    start_index = all_columns.index('Date')
    end_index = all_columns.index('AC') + 1
    match_data = match_data.iloc[:, start_index:end_index]
    # Apply necessary transformations
    match_data['Date'] = pd.to_datetime(match_data['Date'])
    match_data.sort_values(by='Date', inplace=True)
    match_data.reset_index(drop=True, inplace=True)

    # Apply processing functions
    match_data = get_mw(match_data)
    match_data = get_gss(match_data)
    match_data = get_agg_points(match_data)
    match_data = wdl(match_data)
    match_data = add_form_df(match_data)
    match_data = get_last(match_data, standings)

    return match_data


def cumulative_goals_scored(match_data):
    """
    Calculate how many goals each team scored
    """
    goals_scored = {}
    for index, row in match_data.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals, away_goals = row['FTHG'], row['FTAG']

        if home_team not in goals_scored:
            goals_scored[home_team] = [0] * len(match_data)
        if away_team not in goals_scored:
            goals_scored[away_team] = [0] * len(match_data)

        goals_scored[home_team][index] = home_goals
        goals_scored[away_team][index] = away_goals

    for team in goals_scored:
        goals_scored[team] = pd.Series(goals_scored[team]).cumsum()

    df_goals_scored = pd.DataFrame(goals_scored)
    return df_goals_scored


def cumulative_goals_conceded(match_data):
    """
        Calculate how many goals each team conceded
    """
    goals_conceded = {}
    for index, row in match_data.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals_conceded, away_goals_conceded = row['FTAG'], row['FTHG']

        if home_team not in goals_conceded:
            goals_conceded[home_team] = [0] * len(match_data)
        if away_team not in goals_conceded:
            goals_conceded[away_team] = [0] * len(match_data)

        goals_conceded[home_team][index] = home_goals_conceded
        goals_conceded[away_team][index] = away_goals_conceded

    for team in goals_conceded:
        goals_conceded[team] = pd.Series(goals_conceded[team]).cumsum()

    df_goals_conceded = pd.DataFrame(goals_conceded)
    return df_goals_conceded


def get_gss(match_data):
    """
    Load cumulative goals scored/conceded and
    append to goals scored/conceded by home and away teams
    """
    GS = cumulative_goals_scored(match_data)
    GC = cumulative_goals_conceded(match_data)

    HTGS = [GS.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(match_data) + 1), match_data['HomeTeam'])]
    ATGS = [GS.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(match_data) + 1), match_data['AwayTeam'])]
    HTGC = [GC.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(match_data) + 1), match_data['HomeTeam'])]
    ATGC = [GC.loc[:i - 2, team].iloc[-1] if i > 1 else 0 for i, team in
            zip(range(1, len(match_data) + 1), match_data['AwayTeam'])]

    match_data['HTGS'] = HTGS
    match_data['ATGS'] = ATGS
    match_data['HTGC'] = HTGC
    match_data['ATGC'] = ATGC

    return match_data


def get_points(result):
    """
    Calculate point of each match result
    """
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    elif result == 'L':
        return 0


def get_mw(match_data):
    """
    Calculate the match week (Premier League has 20 teams and 38 rounds each season)
    """
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    match_data['MW'] = MatchWeek
    return match_data


def get_matchres(match_data):
    """
    Determine the match result and append to home and away teams
    """
    matchres = []
    for index, row in match_data.iterrows():
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


def get_cuml_points(matchres):
    """
    Calculate cumulative points from match results
    :param matchres:
    :return:
    """
    matchres['Points'] = matchres['Result'].apply(get_points)
    cuml_points = (matchres.pivot_table(index='Matchweek', columns='Team', values='Points', aggfunc='sum')
                   .fillna(0).cumsum())
    return cuml_points.astype(int)


def get_agg_points(match_data):
    """
    Assign cumulative points to match_data, excluding current match
    """
    matchres = get_matchres(match_data)
    cuml_points = get_cuml_points(matchres)
    HTP, ATP = [], []
    for index, row in match_data.iterrows():
        current_mw = row['MW']
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        if current_mw > 1:
            prev_mw = current_mw - 1
            HTP.append(cuml_points.loc[prev_mw, home_team] if prev_mw in cuml_points.index and home_team in cuml_points.columns else 0)
            ATP.append(cuml_points.loc[prev_mw, away_team] if prev_mw in cuml_points.index and away_team in cuml_points.columns else 0)
        else:
            HTP.append(0)
            ATP.append(0)
    match_data['HTP'] = HTP
    match_data['ATP'] = ATP
    return match_data


def wdl(match_data):
    """
    Cumulative wins, draws, and losses for home and away teams
    """
    match_data['HTW'] = 0
    match_data['ATW'] = 0
    match_data['HTD'] = 0
    match_data['ATD'] = 0
    match_data['HTL'] = 0
    match_data['ATL'] = 0

    team_stats = {team: {'W': 0, 'D': 0, 'L': 0} for team in pd.concat([match_data['HomeTeam'], match_data['AwayTeam']]).unique()}

    for index, row in match_data.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        result = row['FTR']

        match_data.at[index, 'HTW'] = team_stats[home_team]['W']
        match_data.at[index, 'ATW'] = team_stats[away_team]['W']
        match_data.at[index, 'HTD'] = team_stats[home_team]['D']
        match_data.at[index, 'ATD'] = team_stats[away_team]['D']
        match_data.at[index, 'HTL'] = team_stats[home_team]['L']
        match_data.at[index, 'ATL'] = team_stats[away_team]['L']

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
        final_index = match_data[(match_data['HomeTeam'] == team) | (match_data['AwayTeam'] == team)].index[-1]
        match_data.at[final_index, 'HTW' if match_data.at[final_index, 'HomeTeam'] == team else 'ATW'] = \
            team_stats[team]['W']
        match_data.at[final_index, 'HTD' if match_data.at[final_index, 'HomeTeam'] == team else 'ATD'] = \
            team_stats[team]['D']
        match_data.at[final_index, 'HTL' if match_data.at[final_index, 'HomeTeam'] == team else 'ATL'] = \
            team_stats[team]['L']

    return match_data


def get_form(match_data, num):
    """
    The following three functions obtain the form of home/away teams for the last five rounds
    """
    form_final = pd.DataFrame()
    matchres = get_matchres(match_data)
    teams = pd.concat([match_data['HomeTeam'], match_data['AwayTeam']]).unique()

    for team in teams:
        team_matches = matchres[matchres['Team'] == team]
        results = team_matches['Result'].tolist()

        form_list = []
        for i in range(0, len(results)):
            form = ''.join(results[max(0, i - num):i]).ljust(num, 'M')
            form_list.append(form)

        team_form_df = pd.DataFrame({
            'Team': [team] * len(team_matches),
            'Matchweek': team_matches['Matchweek'].tolist(),
            'Form': form_list
        })
        form_final = pd.concat([form_final, team_form_df], ignore_index=True)

    return form_final


def add_form(match_data, num):
    match_data = get_mw(match_data)
    form_df = get_form(match_data, num)
    h, a = [], []

    for index, row in match_data.iterrows():
        matchweek = row['MW']
        home_team, away_team = row['HomeTeam'], row['AwayTeam']

        home_form_entry = form_df[(form_df['Team'] == home_team) & (form_df['Matchweek'] == matchweek)]
        away_form_entry = form_df[(form_df['Team'] == away_team) & (form_df['Matchweek'] == matchweek)]

        home_form = home_form_entry['Form'].values[0] if not home_form_entry.empty else 'M'
        away_form = away_form_entry['Form'].values[0] if not away_form_entry.empty else 'M'

        h.append(home_form)
        a.append(away_form)

    match_data[f'HM{num}'] = h
    match_data[f'AM{num}'] = a

    return match_data


def add_form_df(match_dataistics):
    for num in range(1, 6):
        match_dataistics = add_form(match_dataistics, num)
    return match_dataistics


def get_last(match_data, standings):
    """
    Obtain home/away team's last year rank and append to both teams
    """
    match_data['Date'] = pd.to_datetime(match_data['Date'])
    match_data['Season_End_Year'] = match_data['Date'].apply(lambda x: x.year if x.month > 5 else x.year-1)

    team_to_lp = {}
    for year in match_data['Season_End_Year'].unique():
        previous_season_standings = standings[standings['Season_End_Year'] == year]
        team_to_lp.update(previous_season_standings.set_index('Team')['Rk'].to_dict())

    # Teams missing in the previous year are assigned to a 20th position
    match_data['HomeTeamRk'] = match_data.apply(lambda x: team_to_lp.get(x['HomeTeam'], 20), axis=1)
    match_data['AwayTeamRk'] = match_data.apply(lambda x: team_to_lp.get(x['AwayTeam'], 20), axis=1)

    match_data['HomeTeamRk'] = match_data['HomeTeamRk'].astype(int)
    match_data['AwayTeamRk'] = match_data['AwayTeamRk'].astype(int)

    return match_data


def form_and_streaks(seasons_data):
    """
    Determine win/lose streak with the form stats
    """
    match_data = pd.concat(seasons_data.values()).reset_index(drop=True)

    for i in range(1, 6):
        match_data[f'HM{i}'] = match_data[f'HM{i}'].fillna('M')
        match_data[f'AM{i}'] = match_data[f'AM{i}'].fillna('M')

    # Calculate form points
    def get_form_points(form_str):
        points = {'W': 3, 'D': 1, 'L': 0, 'M': 0}  # 'M' for matches not played yet
        return sum(points[result] for result in form_str if result in points)

    match_data['HomeFormPoints_5round'] = match_data['HM5'].apply(get_form_points)
    match_data['AwayFormPoints_5round'] = match_data['AM5'].apply(get_form_points)

    # Identify streaks
    def get_3game_ws(form_str):
        return 'WWW' in form_str

    def get_5game_ws(form_str):
        return 'WWWWW' in form_str

    def get_3game_ls(form_str):
        return 'LLL' in form_str

    def get_5game_ls(form_str):
        return 'LLLLL' in form_str

    match_data['Home3GameWinStreak'] = match_data['HM5'].apply(get_3game_ws).astype(int)
    match_data['Away3GameWinStreak'] = match_data['AM5'].apply(get_3game_ws).astype(int)
    match_data['Home5GameWinStreak'] = match_data['HM5'].apply(get_5game_ws).astype(int)
    match_data['Away5GameWinStreak'] = match_data['AM5'].apply(get_5game_ws).astype(int)
    match_data['Home3GameLossStreak'] = match_data['HM5'].apply(get_3game_ls).astype(int)
    match_data['Away3GameLossStreak'] = match_data['AM5'].apply(get_3game_ls).astype(int)
    match_data['Home5GameLossStreak'] = match_data['HM5'].apply(get_5game_ls).astype(int)
    match_data['Away5GameLossStreak'] = match_data['AM5'].apply(get_5game_ls).astype(int)

    return match_data


def stat_difference(match_data):
    """
    Calculate goal, points, rank difference between home/away teams
    """
    match_data['HTGD'] = match_data['HTGS'] - match_data['HTGC']
    match_data['ATGD'] = match_data['ATGS'] - match_data['ATGC']

    match_data['DiffPts'] = match_data['HTP'] - match_data['ATP']
    match_data['DiffFormPts'] = match_data['HomeFormPoints_5round'] - match_data['AwayFormPoints_5round']

    match_data['DiffLP'] = match_data['HomeTeamRk'] - match_data['AwayTeamRk']

    return match_data


def normalise_data(match_data):
    """
    Normalise statistics
    """
    # Normalise goal statistics columns by their maximum values
    for col in ['HTGS', 'ATGS', 'HTGC', 'ATGC']:
        max_value = match_data[col].max()
        if max_value != 0:  # Avoid division by zero
            match_data[col + '_norm'] = match_data[col] / max_value

    # Normalize goal and points differences by the highest observed absolute value for the matchweek
    for col in ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']:
        # Create a new column for normalized values
        match_data[col] = match_data[col].astype(float)
        match_data[col + '_norm'] = 0.0

        # Loop through each matchweek to normalise based on matchweek data
        for mw in match_data['MW'].unique():
            mw_mask = match_data['MW'] == mw
            max_abs_diff = match_data.loc[mw_mask, col].abs().max()
            if max_abs_diff != 0:  # Avoid division by zero
                match_data.loc[mw_mask, col + '_norm'] = match_data.loc[mw_mask, col] / max_abs_diff

    # Normalize cumulative points by the maximum possible points so far
    match_data['HTP_norm'] = match_data.apply(lambda x: x['HTP'] / ((x['MW'] - 1) * 3) if x['MW'] > 1 else 0, axis=1)
    match_data['ATP_norm'] = match_data.apply(lambda x: x['ATP'] / ((x['MW'] - 1) * 3) if x['MW'] > 1 else 0, axis=1)

    return match_data


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

match_data = form_and_streaks(seasons_data)
match_data = stat_difference(match_data)
match_data = normalise_data(match_data)
match_data_train = match_data[:3800]
match_data_test = match_data[3800:]

match_data_train.to_csv('./Datasets/train_set.csv', index=False)
match_data_test.to_csv('./Datasets/test_set.csv', index=False)


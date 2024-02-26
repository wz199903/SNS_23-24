import pandas as pd


def load_and_process_season(season_suffix, standings):
    """
    Load a season's data, process it, and return the enhanced DataFrame.
    """
    data_path = f'./Datasets/season-{season_suffix}_1.csv'
    playing_stat = pd.read_csv(data_path)
    if 'Time' in playing_stat.columns:
        playing_stat.drop('Time', axis=1, inplace=True)

    all_columns = playing_stat.columns.tolist()
    start_index = all_columns.index('Date')
    end_index = all_columns.index('AR') + 1
    playing_stat = playing_stat.iloc[:, start_index:end_index]
    # Apply necessary transformations
    playing_stat['Date'] = pd.to_datetime(playing_stat['Date'], dayfirst=True)
    playing_stat.sort_values(by='Date', inplace=True)
    playing_stat.reset_index(drop=True, inplace=True)

    # Apply processing functions
    playing_stat = get_mw(playing_stat)
    playing_stat = get_gss(playing_stat)
    playing_stat = get_agg_points(playing_stat)
    playing_stat = wdl(playing_stat)
    playing_stat = add_form_df(playing_stat)
    playing_stat = get_last(playing_stat, standings, int('20' + season_suffix[:2]) + 1)

    return playing_stat


def cumulative_goals_scored(playing_stat):
    goals_scored = {}
    for index, row in playing_stat.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals, away_goals = row['FTHG'], row['FTAG']

        if home_team not in goals_scored:
            goals_scored[home_team] = [home_goals]
        else:
            goals_scored[home_team].append(home_goals + goals_scored[home_team][-1])

        if away_team not in goals_scored:
            goals_scored[away_team] = [away_goals]
        else:
            goals_scored[away_team].append(away_goals + goals_scored[away_team][-1])

    df_goals_scored = pd.DataFrame(goals_scored).ffill().fillna(0)
    df_goals_scored.index += 1
    return df_goals_scored


def cumulative_goals_conceded(playing_stat):
    goals_conceded = {}
    for index, row in playing_stat.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals_conceded, away_goals_conceded = row['FTAG'], row['FTHG']

        if home_team not in goals_conceded:
            goals_conceded[home_team] = [home_goals_conceded]
        else:
            goals_conceded[home_team].append(home_goals_conceded + goals_conceded[home_team][-1])

        if away_team not in goals_conceded:
            goals_conceded[away_team] = [away_goals_conceded]
        else:
            goals_conceded[away_team].append(away_goals_conceded + goals_conceded[away_team][-1])

    df_goals_conceded = pd.DataFrame(goals_conceded).ffill().fillna(0)
    df_goals_conceded.index += 1
    return df_goals_conceded


def get_gss(playing_stat):
    GS = cumulative_goals_scored(playing_stat)
    GC = cumulative_goals_conceded(playing_stat)

    HTGS, ATGS, HTGC, ATGC = [], [], [], []

    for index, row in playing_stat.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        matchweek = index + 1

        htgs = GS[home_team].iloc[matchweek - 1] if home_team in GS and matchweek <= GS.shape[0] else \
            GS[home_team].iloc[-1] if home_team in GS else 0
        atgs = GS[away_team].iloc[matchweek - 1] if away_team in GS and matchweek <= GS.shape[0] else \
            GS[away_team].iloc[-1] if away_team in GS else 0
        htgc = GC[home_team].iloc[matchweek - 1] if home_team in GC and matchweek <= GC.shape[0] else \
            GC[home_team].iloc[-1] if home_team in GC else 0
        atgc = GC[away_team].iloc[matchweek - 1] if away_team in GC and matchweek <= GC.shape[0] else \
            GC[away_team].iloc[-1] if away_team in GC else 0

        HTGS.append(htgs)
        ATGS.append(atgs)
        HTGC.append(htgc)
        ATGC.append(atgc)

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
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_goals, away_goals = row['FTHG'], row['FTAG']

        if home_goals > away_goals:
            matchres.append({'Team': home_team, 'Result': 'W', 'Matchweek': index+1})
            matchres.append({'Team': away_team, 'Result': 'L', 'Matchweek': index + 1})
        elif home_goals < away_goals:
            matchres.append({'Team': home_team, 'Result': 'L', 'Matchweek': index + 1})
            matchres.append({'Team': away_team, 'Result': 'W', 'Matchweek': index + 1})
        else:
            matchres.append({'Team': home_team, 'Result': 'D', 'Matchweek': index + 1})
            matchres.append({'Team': away_team, 'Result': 'D', 'Matchweek': index + 1})

    return pd.DataFrame(matchres)


def get_cuml_points(matchres):
    matchres['Points'] = matchres['Result'].apply(get_points)
    cuml_points = (matchres.pivot_table(index='Matchweek', columns='Team', values='Points', aggfunc='sum')
                   .fillna(0).cumsum())
    return cuml_points.astype(int)


def get_agg_points(playing_stat):
    """Compute and assign cumulative points for each team in the playing_stat DataFrame."""
    matchres = get_matchres(playing_stat)
    cuml_points = get_cuml_points(matchres)

    HTP, ATP = [], []
    for index, row in playing_stat.iterrows():
        matchweek = index + 1
        home_team, away_team = row['HomeTeam'], row['AwayTeam']

        HTP.append(cuml_points.loc[
                       matchweek, home_team] if matchweek in cuml_points.index and home_team in cuml_points.columns else 0)
        ATP.append(cuml_points.loc[
                       matchweek, away_team] if matchweek in cuml_points.index and away_team in cuml_points.columns else 0)

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
    matchres = get_matchres(playing_stat)
    teams = pd.concat([playing_stat['HomeTeam'], playing_stat['AwayTeam']]).unique()

    for team in teams:
        team_matches = matchres[matchres['Team'] == team]
        results = team_matches['Result'].tolist()

        form_list = []
        for i in range(1, len(results) + 1):
            # Generate the form string for each match week, using available results
            form = ''.join(results[max(0, i - num):i]).ljust(num, 'M')
            form_list.append(form)

        team_form_df = pd.DataFrame({
            'Team': [team] * len(team_matches),
            'Matchweek': team_matches['Matchweek'].tolist(),
            'Form': form_list
        })

        form_final = pd.concat([form_final, team_form_df], ignore_index=True)

    return form_final.sort_values(by=['Team', 'Matchweek'])


def add_form(playing_stat, num):
    form_df = get_form(playing_stat, num)
    h, a = [], []  # Initialize lists for home and away form

    # Iterate through each match in playing_stat to assign form
    for index, row in playing_stat.iterrows():
        matchweek = index + 1
        home_team, away_team = row['HomeTeam'], row['AwayTeam']

        # Fetch the home and away form
        home_form_entry = form_df[(form_df['Team'] == home_team) & (form_df['Matchweek'] == matchweek)]
        away_form_entry = form_df[(form_df['Team'] == away_team) & (form_df['Matchweek'] == matchweek)]

        home_form = home_form_entry['Form'].values[0] if not home_form_entry.empty else 'M' * num
        away_form = away_form_entry['Form'].values[0] if not away_form_entry.empty else 'M' * num

        h.append(home_form)
        a.append(away_form)

    # Add the form columns to the playing_stat DataFrame
    playing_stat[f'HomeForm_{num}'] = h
    playing_stat[f'AwayForm_{num}'] = a
    return playing_stat


def add_form_df(playing_statistics):
    for num in range(1, 6):  # Adding form features for 1 to 5 match weeks
        playing_statistics = add_form(playing_statistics, num)
    return playing_statistics


def get_last(playing_stat, standings, year):
    # Last year rankings
    previous_season_standings = standings[standings['Season_End_Year'] == year]

    team_to_lp = previous_season_standings.set_index('Team')['Rk'].to_dict()

    playing_stat['HomeTeamRk'] = playing_stat['HomeTeam'].map(team_to_lp).fillna(20)
    playing_stat['AwayTeamRk'] = playing_stat['AwayTeam'].map(team_to_lp).fillna(20)

    playing_stat['HomeTeamRk'] = playing_stat['HomeTeamRk'].astype(int)
    playing_stat['AwayTeamRk'] = playing_stat['AwayTeamRk'].astype(int)
    return playing_stat


def form_and_streaks(seasons_data):
    playing_stat = pd.concat(seasons_data.values())

    def get_form_points(form_str):
        points = {'W': 3, 'D': 1, 'L': 0}
        return sum(points[result] for result in form_str if result in points)

    # Calculate form points
    playing_stat['HomeFormPoints'] = playing_stat['HomeForm_5'].apply(get_form_points)
    playing_stat['AwayFormPoints'] = playing_stat['AwayForm_5'].apply(get_form_points)

    # Identify streaks
    def get_3game_ws(form_str):
        """Identify if there's a 3-game win streak."""
        return 'WWW' in form_str

    def get_5game_ws(form_str):
        """Identify if there's a 5-game win streak."""
        return 'WWWWW' in form_str

    def get_3game_ls(form_str):
        """Identify if there's a 3-game loss streak."""
        return 'LLL' in form_str

    def get_5game_ls(form_str):
        """Identify if there's a 5-game loss streak."""
        return 'LLLLL' in form_str

    playing_stat['Home3GameWinStreak'] = playing_stat['HomeForm_5'].apply(get_3game_ws)
    playing_stat['Away3GameWinStreak'] = playing_stat['AwayForm_5'].apply(get_3game_ws)
    playing_stat['Home5GameWinStreak'] = playing_stat['HomeForm_5'].apply(get_5game_ws)
    playing_stat['Away5GameWinStreak'] = playing_stat['AwayForm_5'].apply(get_5game_ws)
    playing_stat['Home3GameLossStreak'] = playing_stat['HomeForm_5'].apply(get_3game_ls)
    playing_stat['Away3GameLossStreak'] = playing_stat['AwayForm_5'].apply(get_3game_ls)
    playing_stat['Home5GameLossStreak'] = playing_stat['HomeForm_5'].apply(get_5game_ls)
    playing_stat['Away5GameLossStreak'] = playing_stat['AwayForm_5'].apply(get_5game_ls)

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

    # Normalize goal and points differences by the highest observed absolute value
    for col in ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']:
        max_abs_diff = playing_stat[col].abs().max()
        if max_abs_diff != 0:  # Avoid division by zero
            playing_stat[col + '_norm'] = playing_stat[col] / max_abs_diff

    # Normalize cumulative points by the maximum possible points so far
    # DataFrame is sorted by date and each team plays once per matchweek
    max_points_so_far = (playing_stat.index + 1) * 3
    playing_stat['HTP_norm'] = playing_stat['HTP'] / max_points_so_far
    playing_stat['ATP_norm'] = playing_stat['ATP'] / max_points_so_far

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

print(season_2013_2014_data.head())
print(playing_stat.columns)
playing_stat_train.to_csv('train_set.csv', index=False)
playing_stat_test.to_csv('test_set.csv', index=False)


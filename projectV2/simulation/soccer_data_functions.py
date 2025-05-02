import pandas as pd

def get_team_players(team_name, players_file='players.csv'):
    """
    Extract a dataframe of players from a specific team.

    Parameters:
    -----------
    team_name : str
        The name of the team to filter players by
    players_file : str, optional
        Path to the CSV file containing player data

    Returns:
    --------
    pandas.DataFrame
        Dataframe containing only players from the specified team
    """
    # Read the players CSV file
    players_df = pd.read_csv(players_file)

    # Filter the dataframe to only include players from the specified team
    team_players = players_df[players_df['team'] == team_name]

    return team_players

def get_top_teams(n=32, teams_file='teams.csv', team_styles_file='team_styles.csv'):
    """
    Extract the top N teams based on their level.

    Parameters:
    -----------
    n : int, optional
        Number of top teams to return (default: 32)
    teams_file : str, optional
        Path to the CSV file containing team data
    team_styles_file : str, optional
        Path to the CSV file containing team style data

    Returns:
    --------
    pandas.DataFrame
        Dataframe containing the top N teams sorted by team_level
    """
    # Read the CSV files
    teams_df = pd.read_csv(teams_file)
    team_styles_df = pd.read_csv(team_styles_file)

    # Merge the two dataframes on team ID
    merged_df = pd.merge(
        teams_df,
        team_styles_df,
        left_on='id',
        right_on='team_id',
        how='inner'
    )

    # Sort by team_level in descending order and take the top n teams
    top_teams = merged_df.sort_values(by='team_level', ascending=False).head(n)

    # Select only the columns we want to return
    result_df = top_teams[['id_x', 'name', 'team_level', 'formation', 'possession_style', 'defensive_style']]

    # Rename id_x to id for clarity
    result_df = result_df.rename(columns={'id_x': 'id'})

    return result_df

# Example usage
if __name__ == "__main__":
    # Get all players from a specific team
    arsenal_players = get_team_players("Arsenal")
    print(f"Arsenal players: {len(arsenal_players)}")
    print(arsenal_players.head())

    # Get the top 32 teams
    top_teams = get_top_teams()
    print(f"\nTop 32 teams by level:")
    print(top_teams.head(10))  # Print first 10 of the top 32 teams

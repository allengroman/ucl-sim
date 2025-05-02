import pandas as pd
import matplotlib.pyplot as plt
from soccer_data_functions import get_team_players, get_top_teams

def main():
    # Set display options for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Example 1: Get players from a specific team
    team_name = "Manchester City"
    print(f"\n--- Players from {team_name} ---")
    team_players = get_team_players(team_name)
    print(f"Number of players: {len(team_players)}")
    print(team_players[['name', 'position', 'pace', 'stamina', 'finishing']].head(10))

    # Example 2: Get the top 32 teams by level
    print("\n--- Top 32 Teams by Level ---")
    top_teams = get_top_teams(32)
    print(top_teams.head(10))  # Show only first 10 for brevity

    # Example 3: Compare player attributes across positions for a team
    print(f"\n--- Position Analysis for {team_name} ---")
    positions = team_players['position'].unique()

    # Calculate average attributes by position
    position_stats = team_players.groupby('position').agg({
        'pace': 'mean',
        'stamina': 'mean',
        'strength': 'mean',
        'short_passing': 'mean',
        'finishing': 'mean',
        'tackling': 'mean'
    }).round(1)

    print(position_stats)

    # Example 4: Find the most balanced teams (teams with high levels and balanced styles)
    print("\n--- Most Balanced Teams ---")
    # Get all teams with their styles
    all_teams = get_top_teams(n=96)  # Get all teams

    # Filter teams with balanced possession style and balanced attacking mindset
    balanced_teams = all_teams[
        (all_teams['possession_style'] == 'Balanced') &
        (all_teams['defensive_style'].isin(['Balanced', 'Mid-Block']))
    ]

    # Sort by team level
    balanced_teams = balanced_teams.sort_values(by='team_level', ascending=False)
    print(balanced_teams.head(5))

    # Example 5: Get top players from a position across all teams
    print("\n--- Top Midfielders by Short Passing ---")
    # Read all players
    all_players = pd.read_csv('players.csv')

    # Filter midfielders and sort by short passing
    midfielders = all_players[all_players['position'] == 'MF']
    top_passers = midfielders.sort_values(by='short_passing', ascending=False)
    print(top_passers[['name', 'team', 'short_passing', 'vision']].head(10))

if __name__ == "__main__":
    main()

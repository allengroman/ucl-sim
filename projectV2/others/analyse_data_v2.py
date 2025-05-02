import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from soccer_data_functions import get_team_players, get_top_teams

class SoccerAnalyzer:
    def __init__(self, players_file='players.csv', teams_file='teams.csv', team_styles_file='team_styles.csv'):
        """Initialize the analyzer with data files."""
        self.players_file = players_file
        self.teams_file = teams_file
        self.team_styles_file = team_styles_file

        # Load all data
        self.players_df = pd.read_csv(players_file)
        self.teams_df = pd.read_csv(teams_file)
        self.team_styles_df = pd.read_csv(team_styles_file)

        # Create a merged dataframe of teams and styles
        self.teams_with_styles = pd.merge(
            self.teams_df,
            self.team_styles_df,
            left_on='id',
            right_on='team_id',
            how='inner'
        )

        # Rename columns for clarity
        self.teams_with_styles = self.teams_with_styles.rename(columns={'id_x': 'id'})

    def get_team_players(self, team_name):
        """Get all players for a specific team."""
        return get_team_players(team_name, self.players_file)

    def get_top_teams(self, n=32):
        """Get top teams by level."""
        return get_top_teams(n, self.teams_file, self.team_styles_file)

    def team_radar_chart(self, team_name):
        """
        Create a radar chart of team players' average attributes.
        Returns the attributes dictionary for further processing.
        """
        team_players = self.get_team_players(team_name)

        # Calculate average attributes for the team
        attributes = {
            'Pace': team_players['pace'].mean(),
            'Stamina': team_players['stamina'].mean(),
            'Strength': team_players['strength'].mean(),
            'Short Passing': team_players['short_passing'].mean(),
            'Long Passing': team_players['long_passing'].mean(),
            'Vision': team_players['vision'].mean(),
            'Tackling': team_players['tackling'].mean(),
            'Finishing': team_players['finishing'].mean()
        }

        # Convert to list for plotting
        attributes_list = list(attributes.values())
        attributes_names = list(attributes.keys())

        # Plotting code would go here if you want to visualize
        # For now, just return the attributes
        return attributes

    def compare_teams(self, team_names):
        """Compare multiple teams based on player attributes."""
        results = {}

        for team in team_names:
            results[team] = self.team_radar_chart(team)

        # Return a dataframe for easy comparison
        return pd.DataFrame(results)

    def get_best_formation(self, player_attributes):
        """
        Recommend the best formation based on player attributes.

        Parameters:
        player_attributes: dict of player attributes

        Returns:
        str: Recommended formation
        """
        # Simple logic - could be expanded with more complex algorithms
        attack_strength = (player_attributes.get('Pace', 0) +
                          player_attributes.get('Finishing', 0) +
                          player_attributes.get('Vision', 0)) / 3

        midfield_strength = (player_attributes.get('Short Passing', 0) +
                            player_attributes.get('Long Passing', 0) +
                            player_attributes.get('Stamina', 0)) / 3

        defense_strength = (player_attributes.get('Tackling', 0) +
                           player_attributes.get('Strength', 0)) / 2

        # Recommend formation based on relative strengths
        if attack_strength > midfield_strength and attack_strength > defense_strength:
            # Attack-oriented formation
            return "4-3-3" if midfield_strength > defense_strength else "3-4-3"
        elif midfield_strength > attack_strength and midfield_strength > defense_strength:
            # Midfield-oriented formation
            return "4-5-1" if defense_strength > attack_strength else "4-2-3-1"
        else:
            # Defense-oriented formation
            return "5-3-2" if midfield_strength > attack_strength else "5-4-1"

    def find_similar_teams(self, team_name, n=5):
        """Find teams with similar playing styles."""
        # Get team style data
        team_style = self.teams_with_styles[self.teams_with_styles['name'] == team_name].iloc[0]

        # Teams with same formation
        same_formation = self.teams_with_styles[
            (self.teams_with_styles['formation'] == team_style['formation']) &
            (self.teams_with_styles['name'] != team_name)
        ]

        # Teams with same playing style
        same_style = same_formation[
            (same_formation['possession_style'] == team_style['possession_style']) |
            (same_formation['defensive_style'] == team_style['defensive_style'])
        ]

        # Sort by team level similarity
        same_style['level_diff'] = abs(same_style['team_level'] - team_style['team_level'])
        similar_teams = same_style.sort_values(by='level_diff').head(n)

        return similar_teams[['name', 'team_level', 'formation', 'possession_style', 'defensive_style']]

    def find_transfer_targets(self, team_name, position, min_rating=70, max_age=30):
        """
        Find potential transfer targets for a team based on position and attributes.

        In a real scenario, this would include player age, value, etc.
        This simplified version just looks at attributes.
        """
        # Here we're simulating age - in reality, this would be in your data
        # Current team players
        team_players = self.get_team_players(team_name)
        team_player_ids = team_players['id'].tolist()

        # Find players from other teams with the same position
        potential_targets = self.players_df[
            (self.players_df['position'] == position) &
            (~self.players_df['id'].isin(team_player_ids))
        ]

        # Calculate a simple rating based on key attributes for the position
        if position == 'DF':
            # For defenders
            potential_targets['rating'] = (
                potential_targets['tackling'] * 0.3 +
                potential_targets['strength'] * 0.3 +
                potential_targets['interceptions'] * 0.2 +
                potential_targets['pace'] * 0.1 +
                potential_targets['short_passing'] * 0.1
            )
        elif position == 'MF':
            # For midfielders
            potential_targets['rating'] = (
                potential_targets['short_passing'] * 0.3 +
                potential_targets['vision'] * 0.2 +
                potential_targets['stamina'] * 0.2 +
                potential_targets['ball_control'] * 0.2 +
                potential_targets['long_passing'] * 0.1
            )
        elif position == 'FW':
            # For forwards
            potential_targets['rating'] = (
                potential_targets['finishing'] * 0.3 +
                potential_targets['pace'] * 0.2 +
                potential_targets['dribbling'] * 0.2 +
                potential_targets['shot_power'] * 0.2 +
                potential_targets['ball_control'] * 0.1
            )
        else:
            # For goalkeepers or others
            potential_targets['rating'] = (
                potential_targets['reflexes'] * 0.4 +
                potential_targets['handling'] * 0.3 +
                potential_targets['gk_positioning'] * 0.3
            )

        # Filter by minimum rating and sort
        filtered_targets = potential_targets[potential_targets['rating'] >= min_rating]
        sorted_targets = filtered_targets.sort_values(by='rating', ascending=False)

        # Return the top potential targets
        return sorted_targets[['name', 'team', 'rating', 'pace', 'stamina', 'strength',
                             'short_passing', 'finishing', 'tackling']]

# Example usage
def main():
    # Create the analyzer
    analyzer = SoccerAnalyzer()

    # Get top teams
    top_teams = analyzer.get_top_teams(10)
    print("Top 10 Teams:")
    print(top_teams[['name', 'team_level', 'formation']])
    print()

    # Choose a team for detailed analysis
    team_name = "Liverpool"
    print(f"Analyzing {team_name}...")

    # Get team players
    team_players = analyzer.get_team_players(team_name)
    print(f"Number of players: {len(team_players)}")
    print(team_players[['name', 'position', 'pace', 'stamina', 'finishing']].head(5))
    print()

    # Get team attributes and recommend formation
    attributes = analyzer.team_radar_chart(team_name)
    print("Team Attributes:")
    for attr, value in attributes.items():
        print(f"{attr}: {value:.1f}")
    print()

    recommended_formation = analyzer.get_best_formation(attributes)
    print(f"Recommended Formation: {recommended_formation}")
    print()

    # Find similar teams
    similar_teams = analyzer.find_similar_teams(team_name)
    print(f"Teams similar to {team_name}:")
    print(similar_teams)
    print()

    # Compare with a rival
    rival_team = "Barcelona"
    comparison = analyzer.compare_teams([team_name, rival_team])
    print(f"Comparison with {rival_team}:")
    print(comparison)
    print()

    # Find transfer targets for a position
    position = "MF"
    print(f"Potential {position} transfer targets for {team_name}:")
    targets = analyzer.find_transfer_targets(team_name, position)
    print(targets.head(5))

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np

def get_top_eleven(team_name=None, formation="4-3-3", players_file='players.csv', teams_file='teams.csv', team_styles_file='team_styles.csv'):
    """
    Extract the top 11 players from a team or league based on a specified formation.

    Parameters:
    -----------
    team_name : str, optional
        The name of the team to filter players by. If None, selects from all players.
    formation : str, optional
        The formation to use for selecting players (e.g., "4-3-3", "4-4-2", "3-5-2")
    players_file : str, optional
        Path to the CSV file containing player data
    teams_file : str, optional
        Path to the CSV file containing team data
    team_styles_file : str, optional
        Path to the CSV file containing team style data

    Returns:
    --------
    dict
        Dictionary containing DataFrames for each position group (GK, DF, MF, FW)
        and the complete best eleven as a single DataFrame
    """
    try:
        # Check if files exist and handle paths
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths if relative paths were provided
        if not os.path.isabs(players_file):
            players_file = os.path.join(current_dir, players_file)
            
        # Read the players CSV file
        players_df = pd.read_csv(players_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the players file: {players_file}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The players file is empty: {players_file}")
    except pd.errors.ParserError:
        raise ValueError(f"Could not parse the players file: {players_file}. Check the CSV format.")

    # Filter by team if specified
    if team_name:
        players_df = players_df[players_df['team'] == team_name]
        if len(players_df) == 0:
            raise ValueError(f"No players found for team: {team_name}")

    # Parse the formation to get the number of players in each position
    positions = formation.split('-')
    if len(positions) < 3:
        raise ValueError(f"Invalid formation format: {formation}. Expected format like '4-3-3'")

    num_defenders = int(positions[0])
    num_midfielders = int(positions[1])
    num_forwards = int(positions[2])

    # Create position-specific rating functions
    def rate_goalkeeper(player):
        return (
            player['reflexes'] * 0.4 +
            player['handling'] * 0.3 +
            player['gk_positioning'] * 0.3
        )

    def rate_defender(player):
        return (
            player['tackling'] * 0.25 +
            player['interceptions'] * 0.2 +
            player['strength'] * 0.2 +
            player['blocking'] * 0.15 +
            player['pace'] * 0.1 +
            player['positioning'] * 0.1
        )

    def rate_midfielder(player):
        return (
            player['short_passing'] * 0.25 +
            player['vision'] * 0.2 +
            player['stamina'] * 0.15 +
            player['ball_control'] * 0.15 +
            player['long_passing'] * 0.15 +
            player['interceptions'] * 0.1
        )

    def rate_forward(player):
        return (
            player['finishing'] * 0.3 +
            player['shot_power'] * 0.2 +
            player['pace'] * 0.15 +
            player['dribbling'] * 0.15 +
            player['ball_control'] * 0.1 +
            player['positioning'] * 0.1
        )

    # Filter players by position and add ratings
    # Map positions in the dataset to standard format
    position_map = {
        'GK': 'GK',  # Goalkeeper
        'DF': 'DF',  # Defender
        'MD': 'MF',  # Midfielder (in dataset as 'MD')
        'MF': 'MF',  # Midfielder (standard)
        'FW': 'FW',  # Forward
        'ST': 'FW'   # Striker (alternative name for forward)
    }
    
    # Add a standardized position column
    players_df['std_position'] = players_df['position'].map(position_map).fillna('MF')
    
    # Filter using the standard positions
    goalkeepers = players_df[players_df['std_position'] == 'GK'].copy()
    defenders = players_df[players_df['std_position'] == 'DF'].copy()
    midfielders = players_df[players_df['std_position'] == 'MF'].copy()
    forwards = players_df[players_df['std_position'] == 'FW'].copy()
    
    # Check if we have enough players for the formation
    if len(goalkeepers) < 1:
        # If no goalkeepers, create a dummy goalkeeper
        print(f"Warning: No goalkeepers found. Creating a dummy goalkeeper.")
        dummy_gk = players_df.iloc[0:1].copy()
        dummy_gk['std_position'] = 'GK'
        dummy_gk['position'] = 'GK'
        dummy_gk['name'] = dummy_gk['name'] + " (GK)"
        goalkeepers = pd.concat([goalkeepers, dummy_gk])
    
    # Check if we have enough defenders
    if len(defenders) < num_defenders:
        print(f"Warning: Not enough defenders ({len(defenders)}) for formation ({num_defenders} needed). Using midfielders as defenders.")
        # Use some midfielders as defenders
        mids_as_defs = midfielders.iloc[:(num_defenders - len(defenders))].copy()
        mids_as_defs['std_position'] = 'DF'
        mids_as_defs['position'] = 'DF'
        mids_as_defs['name'] = mids_as_defs['name'] + " (DF)"
        defenders = pd.concat([defenders, mids_as_defs])
    
    # Check if we have enough midfielders
    if len(midfielders) < num_midfielders:
        print(f"Warning: Not enough midfielders ({len(midfielders)}) for formation ({num_midfielders} needed). Using forwards or defenders as midfielders.")
        # First try to use forwards, then defenders
        needed = num_midfielders - len(midfielders)
        if len(forwards) > num_forwards:
            # Use some forwards as midfielders
            fws_as_mids = forwards.iloc[:min(needed, len(forwards) - num_forwards)].copy()
            fws_as_mids['std_position'] = 'MF'
            fws_as_mids['position'] = 'MF'
            fws_as_mids['name'] = fws_as_mids['name'] + " (MF)"
            midfielders = pd.concat([midfielders, fws_as_mids])
            needed -= len(fws_as_mids)
        
        if needed > 0 and len(defenders) > num_defenders:
            # Use some defenders as midfielders
            defs_as_mids = defenders.iloc[:min(needed, len(defenders) - num_defenders)].copy()
            defs_as_mids['std_position'] = 'MF'
            defs_as_mids['position'] = 'MF'
            defs_as_mids['name'] = defs_as_mids['name'] + " (MF)"
            midfielders = pd.concat([midfielders, defs_as_mids])
    
    # Check if we have enough forwards
    if len(forwards) < num_forwards:
        print(f"Warning: Not enough forwards ({len(forwards)}) for formation ({num_forwards} needed). Using midfielders as forwards.")
        # Use some midfielders as forwards
        mids_as_fws = midfielders.iloc[:(num_forwards - len(forwards))].copy()
        mids_as_fws['std_position'] = 'FW'
        mids_as_fws['position'] = 'FW'
        mids_as_fws['name'] = mids_as_fws['name'] + " (FW)"
        forwards = pd.concat([forwards, mids_as_fws])

    # Calculate ratings for each position
    goalkeepers['rating'] = goalkeepers.apply(rate_goalkeeper, axis=1)
    defenders['rating'] = defenders.apply(rate_defender, axis=1)
    midfielders['rating'] = midfielders.apply(rate_midfielder, axis=1)
    forwards['rating'] = forwards.apply(rate_forward, axis=1)

    # Sort players by rating and select the top N for each position
    top_gk = goalkeepers.sort_values('rating', ascending=False).head(1)
    top_df = defenders.sort_values('rating', ascending=False).head(num_defenders)
    top_mf = midfielders.sort_values('rating', ascending=False).head(num_midfielders)
    top_fw = forwards.sort_values('rating', ascending=False).head(num_forwards)

    # Add position ranks before concatenating
    # Goalkeepers (always just one)
    top_gk = top_gk.copy()
    top_gk['position_rank'] = 1
    top_gk['detailed_position'] = 'GK1'

    # Defenders
    top_df = top_df.copy()
    top_df['position_rank'] = range(1, len(top_df) + 1)
    # Create detailed_position one by one
    for i, idx in enumerate(top_df.index):
        top_df.loc[idx, 'detailed_position'] = f"DF{i+1}"

    # Midfielders
    top_mf = top_mf.copy()
    top_mf['position_rank'] = range(1, len(top_mf) + 1)
    # Create detailed_position one by one
    for i, idx in enumerate(top_mf.index):
        top_mf.loc[idx, 'detailed_position'] = f"MF{i+1}"

    # Forwards
    top_fw = top_fw.copy()
    top_fw['position_rank'] = range(1, len(top_fw) + 1)
    # Create detailed_position one by one
    for i, idx in enumerate(top_fw.index):
        top_fw.loc[idx, 'detailed_position'] = f"FW{i+1}"

    # Combine into the best eleven
    best_eleven = pd.concat([top_gk, top_df, top_mf, top_fw])

    # Sort by position and then by rating within position
    best_eleven = best_eleven.sort_values(['position', 'position_rank'])

    # Select columns for the output
    columns = ['name', 'position', 'detailed_position', 'team', 'rating']
    attribute_cols = ['pace', 'stamina', 'strength', 'short_passing', 'long_passing',
                     'vision', 'finishing', 'shot_power', 'dribbling', 'ball_control', 
                     'tackling', 'interceptions', 'blocking', 'positioning', 
                     'reflexes', 'handling', 'gk_positioning']

    # Return both the best eleven and separate position groups
    return {
        'best_eleven': best_eleven[columns + attribute_cols],
        'goalkeepers': top_gk[columns + attribute_cols],
        'defenders': top_df[columns + attribute_cols],
        'midfielders': top_mf[columns + attribute_cols],
        'forwards': top_fw[columns + attribute_cols],
        'formation': formation,
        'team': team_name
    }

def display_top_eleven(result):
    """
    Display the results of get_top_eleven in a readable format.

    Parameters:
    -----------
    result : dict
        The dictionary returned by get_top_eleven
    """
    team_name = result['team'] or "All Teams"
    formation = result['formation']

    print(f"\n{'='*50}")
    print(f"Top Eleven for {team_name} - Formation: {formation}")
    print(f"{'='*50}")

    # Display by position group
    for position, label in [
        ('goalkeepers', 'GOALKEEPER'),
        ('defenders', 'DEFENDERS'),
        ('midfielders', 'MIDFIELDERS'),
        ('forwards', 'FORWARDS')
    ]:
        players = result[position]
        print(f"\n{label} ({len(players)} players)")
        print('-' * 30)

        # Format the output with key stats
        for i, (_, player) in enumerate(players.iterrows()):
            # Get position prefix based on the position group
            if position == 'goalkeepers':
                pos_prefix = 'GK'
                reflexes = player['reflexes'] if 'reflexes' in player else 'N/A'
                handling = player['handling'] if 'handling' in player else 'N/A'
                key_stats = f"(Reflexes: {reflexes}, Handling: {handling})"
            elif position == 'defenders':
                pos_prefix = 'DF'
                tackling = player['tackling'] if 'tackling' in player else 'N/A'
                strength = player['strength'] if 'strength' in player else 'N/A'
                key_stats = f"(Tackling: {tackling}, Strength: {strength})"
            elif position == 'midfielders':
                pos_prefix = 'MF'
                passing = player['short_passing'] if 'short_passing' in player else 'N/A'
                vision = player['vision'] if 'vision' in player else 'N/A'
                key_stats = f"(Passing: {passing}, Vision: {vision})"
            else:  # forwards
                pos_prefix = 'FW'
                finishing = player['finishing'] if 'finishing' in player else 'N/A'
                pace = player['pace'] if 'pace' in player else 'N/A'
                key_stats = f"(Finishing: {finishing}, Pace: {pace})"
            
            # Use detailed_position if available, otherwise construct it
            if 'detailed_position' in player:
                pos_display = player['detailed_position']
            else:
                pos_display = f"{pos_prefix}{i+1}"

            print(f"{pos_display}. {player['name']} - Rating: {player['rating']:.1f} {key_stats}")

    print(f"\n{'='*50}")
    print(f"Total Rating: {result['best_eleven']['rating'].sum():.1f}")
    print(f"{'='*50}")


# Example usage
if __name__ == "__main__":
    import os
    
    print("Running top eleven function examples...")
    
    # Example 1: Get the best eleven from all players using a 4-3-3 formation
    try:
        print("\nExample 1: All players with 4-3-3 formation")
        best_overall = get_top_eleven(formation="4-3-3")
        display_top_eleven(best_overall)
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    # Get list of teams from the players data
    try:
        # Read players CSV to get actual teams in the data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        players_file = os.path.join(current_dir, 'players.csv')
        players_df = pd.read_csv(players_file)
        teams = players_df['team'].unique()
        
        if len(teams) > 0:
            # Example 2: Get the best eleven from first team in the data
            team_name = teams[0]
            try:
                print(f"\nExample 2: Team {team_name} with 4-3-3 formation")
                team_best = get_top_eleven(team_name=team_name, formation="4-3-3")
                display_top_eleven(team_best)
            except Exception as e:
                print(f"Error in Example 2: {e}")
                
            # Example 3: Try a different formation (3-5-2) with another team if available
            if len(teams) > 1:
                team_name = teams[1]
            try:
                print(f"\nExample 3: Team {team_name} with 3-5-2 formation")
                different_formation = get_top_eleven(team_name=team_name, formation="3-5-2")
                display_top_eleven(different_formation)
            except Exception as e:
                print(f"Error in Example 3: {e}")
    except Exception as e:
        print(f"Error reading team data: {e}")

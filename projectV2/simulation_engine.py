import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import random
from collections import deque
from enum import Enum
import time
import os

# Import the model architecture from our other module
from soccer_event_prediction_model import EventMultiTaskPredictor, OffBallMovementPredictor

class EventType(Enum):
    """Enum for event types"""
    PASS = "PASS"
    CARRY = "CARRY"
    DUEL = "DUEL"
    BALL_RECOVERY = "BALL RECOVERY"
    SHOT = "SHOT"
    GOAL = "GOAL"
    CLEARANCE = "CLEARANCE"
    INTERCEPTION = "INTERCEPTION"
    FOUL = "FOUL"
    OFFSIDE = "OFFSIDE"
    CORNER = "CORNER"
    FREE_KICK = "FREE KICK"
    THROW_IN = "THROW IN"
    KICKOFF = "KICKOFF"


class SoccerMatchSimulator:
    """
    Main class for soccer match simulation with discrete event-driven approach
    """
    def __init__(self, home_team_id, away_team_id, players_data, teams_data, team_styles_data,
                 event_model=None, movement_model=None):
        """
        Initialize the soccer match simulator
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            players_data: DataFrame with player data
            teams_data: DataFrame with team data
            team_styles_data: DataFrame with team tactical styles data
            event_model: Pre-trained event prediction model (or None to use random simulation)
            movement_model: Pre-trained movement prediction model (or None to use rule-based movements)
        """
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        
        # Load data
        self.players_data = players_data
        self.teams_data = teams_data
        self.team_styles_data = team_styles_data
        
        # Set up models
        self.event_model = event_model
        self.movement_model = movement_model
        
        # Initialize match state
        self.pitch_length = 105  # meters, standard pitch length
        self.pitch_width = 68   # meters, standard pitch width
        
        # Initialize match state
        self.reset_match()
        
    def reset_match(self):
        """Reset the match to its initial state"""
        # Match state
        self.current_time = 0  # seconds from kickoff
        # Ensure exactly equal probability for both teams, avoiding any home advantage bias
        self.possession = self.home_team_id if random.random() < 0.5 else self.away_team_id
        self.score = {self.home_team_id: 0, self.away_team_id: 0}
        self.period = 1  # 1 for first half, 2 for second half
        
        # Event queue for discrete event simulation
        self.event_queue = deque()
        
        # Match history
        self.events = []
        self.player_positions_history = []
        
        # Player selection and positioning
        self.select_players()
        self.initialize_player_positions()
        
        # Ball state
        self.ball_position = [self.pitch_length/2, self.pitch_width/2]  # Start at center
        self.ball_carrier = None
        
        # Schedule kickoff to start the match
        self._schedule_event(EventType.KICKOFF, 0, self.possession)
        
    def select_players(self):
        """Select players for both teams based on attributes"""
        # Get team formations and styles
        home_style = self.team_styles_data[self.team_styles_data['team_id'] == self.home_team_id]
        if home_style.empty:
            print(f"Warning: Team style not found for home team ID {self.home_team_id}. Using default formation.")
            home_formation = [4, 4, 2]  # Default to 4-4-2
            home_style_dict = {'formation': '4-4-2'}
        else:
            home_style_dict = home_style.iloc[0]
            home_formation = self._parse_formation(home_style_dict['formation'])
            
        away_style = self.team_styles_data[self.team_styles_data['team_id'] == self.away_team_id]
        if away_style.empty:
            print(f"Warning: Team style not found for away team ID {self.away_team_id}. Using default formation.")
            away_formation = [4, 4, 2]  # Default to 4-4-2
            away_style_dict = {'formation': '4-4-2'}
        else:
            away_style_dict = away_style.iloc[0]
            away_formation = self._parse_formation(away_style_dict['formation'])
        
        # Filter players for each team - use exact string comparison to ensure correct team matching
        home_players_df = self.players_data[self.players_data['team'] == self.home_team_id]
        away_players_df = self.players_data[self.players_data['team'] == self.away_team_id]
        
        print(f"Found {len(home_players_df)} players for home team (ID: {self.home_team_id})")
        print(f"Found {len(away_players_df)} players for away team (ID: {self.away_team_id})")
        
        # Set current team for player selection
        self.current_team_selection = self.home_team_id
        
        # Select players for home team
        self.home_players = {
            'GK': self._select_best_players(home_players_df, 'GK', 1),
            'DEF': self._select_best_players(home_players_df, 'DF', home_formation[0]),
            'MID': self._select_best_players(home_players_df, 'MD', home_formation[1]),
            'FWD': self._select_best_players(home_players_df, 'AT', home_formation[2])
        }
        
        # Change current team for player selection
        self.current_team_selection = self.away_team_id
        
        # Select players for away team
        self.away_players = {
            'GK': self._select_best_players(away_players_df, 'GK', 1),
            'DEF': self._select_best_players(away_players_df, 'DF', away_formation[0]),
            'MID': self._select_best_players(away_players_df, 'MD', away_formation[1]),
            'FWD': self._select_best_players(away_players_df, 'AT', away_formation[2])
        }
        
        # Reset current team selection
        self.current_team_selection = None
        
        # Combine all players
        self.home_lineup = {}
        for group_name, group in self.home_players.items():
            if group:  # Only add if the group contains players
                for player in group:
                    self.home_lineup[player['id']] = player
        
        self.away_lineup = {}
        for group_name, group in self.away_players.items():
            if group:  # Only add if the group contains players
                for player in group:
                    self.away_lineup[player['id']] = player
        
        # Print player counts
        print(f"Selected {len(self.home_lineup)} players for home team")
        print(f"Selected {len(self.away_lineup)} players for away team")
                    
        # Ensure we have at least some players to work with
        if not self.home_lineup:
            print("Warning: No players available for home team. Using generic players.")
            # Create some generic players for the home team
            for i in range(11):
                self.home_lineup[1000 + i] = {'id': 1000 + i, 'name': f'HomePlayer{i}', 'team': self.home_team_id, 'position': 'Generic'}
        
        if not self.away_lineup:
            print("Warning: No players available for away team. Using generic players.")
            # Create some generic players for the away team
            for i in range(11):
                self.away_lineup[2000 + i] = {'id': 2000 + i, 'name': f'AwayPlayer{i}', 'team': self.away_team_id, 'position': 'Generic'}
        
    def _parse_formation(self, formation_str):
        """Parse formation string into number of players in each position"""
        return [int(x) for x in formation_str.split('-')]
    
    def _select_best_players(self, players_df, positions, count):
        """Select the best players for a given position"""
        if isinstance(positions, str):
            positions = [positions]
            
        # Simple mapping of requested positions to CSV position values    
        position_mapping = {
            'GK': 'GK',
            'CB': 'DF',
            'LB': 'DF',
            'RB': 'DF',
            'CDM': 'MD',
            'CM': 'MD',
            'CAM': 'MD',
            'LM': 'MD',
            'RM': 'MD',
            'LW': 'AT',
            'RW': 'AT',
            'ST': 'AT',
            'CF': 'AT'
        }
        
        # Map requested positions to CSV values
        csv_positions = []
        for pos in positions:
            if pos in position_mapping:
                csv_positions.append(position_mapping[pos])
            else:
                csv_positions.append(pos)  # Keep original if not in mapping
        
        # Debugging information
        print(f"Selecting {count} players for position(s) {csv_positions}")
        print(f"Initial dataframe has {len(players_df)} players")
        
        # First, ensure we're working with the correct team
        if self.current_team_selection is not None:
            team_mask = players_df['team'] == self.current_team_selection
            team_players = players_df[team_mask].copy()
            
            if len(team_players) == 0:
                print(f"WARNING: No players found for team ID {self.current_team_selection}")
                # If no players are found with the exact team ID match, create generic players
                generic_players = []
                for i in range(count):
                    base_id = 1000 if self.current_team_selection == self.home_team_id else 2000
                    generic_players.append({
                        'id': base_id + len(generic_players),
                        'name': f"Generic{csv_positions[0]}{i+1}",
                        'team': self.current_team_selection,
                        'position': csv_positions[0],
                        'position_rating': 50.0  # Average rating
                    })
                return generic_players
            
            # Now filter by position within this team
            position_mask = team_players['position'].isin(csv_positions)
            filtered_players = team_players[position_mask].copy()
            
            print(f"After team and position filtering: {len(filtered_players)} players")
            
            # If we don't have enough players in the exact positions, use players from other positions
            if len(filtered_players) < count:
                # Try any players from the team
                print(f"Warning: Not enough players for position(s) {csv_positions} in team {self.current_team_selection}.")
                print(f"Using {len(team_players)} players from any position in this team.")
                filtered_players = team_players.copy()
        else:
            # No team filtering - just apply position filter
            position_mask = players_df['position'].isin(csv_positions)
            filtered_players = players_df[position_mask].copy()
            
            print(f"After position filtering: {len(filtered_players)} players")
        
        # Verify we have enough players for this position
        if len(filtered_players) < count:
            # Create generic players to fill the gap
            needed = count - len(filtered_players)
            print(f"Warning: Need {needed} more players for position(s) {csv_positions}. Creating generic players.")
            
            existing_players = filtered_players.to_dict('records')
            
            # Create generic players
            for i in range(needed):
                base_id = 1000 if self.current_team_selection == self.home_team_id else 2000
                player_id = base_id + 100 + i  # Avoid ID conflicts
                
                generic_player = {
                    'id': player_id,
                    'name': f"Generic{csv_positions[0]}{i+1}",
                    'team': self.current_team_selection,
                    'position': csv_positions[0],
                    'position_rating': 50.0  # Average rating
                }
                
                # Add attributes that may be needed
                for attr in ['pace', 'stamina', 'strength', 'short_passing', 'vision', 
                           'ball_control', 'tackling', 'interceptions', 'blocking', 
                           'finishing', 'shot_power', 'dribbling', 'gk_positioning']:
                    generic_player[attr] = 50.0  # Average rating
                
                existing_players.append(generic_player)
            
            return existing_players[:count]  # Return exactly the number requested
        
        # For real players, select based on attributes
        # For simplicity, select players with highest overall rating for position
        if 'GK' in csv_positions:
            # For goalkeepers, use goalkeeper-specific attributes that exist in the dataset
            rating_cols = ['gk_positioning']  # Use only the column that exists
        elif any(pos in ['DF'] for pos in csv_positions):
            # For defenders, use defensive attributes
            rating_cols = ['tackling', 'interceptions', 'blocking', 'strength']
        elif any(pos in ['MD'] for pos in csv_positions):
            # For midfielders, use midfield attributes
            rating_cols = ['stamina', 'short_passing', 'vision', 'ball_control']
        else:
            # For forwards, use attacking attributes
            rating_cols = ['pace', 'finishing', 'shot_power', 'dribbling']
        
        # Make sure all the columns we're using actually exist
        available_cols = [col for col in rating_cols if col in filtered_players.columns]
        if not available_cols:
            # If none of the preferred columns exist, fall back to a default column or create one
            print(f"Warning: None of the preferred attribute columns {rating_cols} exist. Creating position_rating column.")
            filtered_players['position_rating'] = 50.0  # Average rating
        else:
            # Calculate overall rating for position - handle missing values by filling with average (50)
            for col in available_cols:
                if filtered_players[col].isna().any():
                    filtered_players[col] = filtered_players[col].fillna(50.0)
            
            filtered_players['position_rating'] = filtered_players[available_cols].mean(axis=1)
        
        # Select top players
        selected_players = filtered_players.sort_values('position_rating', ascending=False).head(count)
        
        # Convert to list of dicts for easier access
        return selected_players.to_dict('records')
    
    def initialize_player_positions(self):
        """Initialize player positions based on team formation"""
        # Get team formations and styles
        try:
            home_style_row = self.team_styles_data[self.team_styles_data['team_id'] == self.home_team_id]
            away_style_row = self.team_styles_data[self.team_styles_data['team_id'] == self.away_team_id]
            
            if home_style_row.empty or away_style_row.empty:
                raise ValueError(f"Team style data not found for teams {self.home_team_id} and/or {self.away_team_id}")
                
            home_style = home_style_row.iloc[0]
            away_style = away_style_row.iloc[0]
            
            home_formation = home_style['formation']
            away_formation = away_style['formation']
        except Exception as e:
            print(f"Warning: Couldn't load team formations: {e}. Using default formations.")
            home_formation = "4-4-2"
            away_formation = "4-4-2"
        
        # Initialize dictionaries for player positions
        self.home_positions = {}
        self.away_positions = {}
        
        # Position home players
        self._position_players(
            self.home_players, 
            self.home_positions, 
            home_formation, 
            is_home=True
        )
        
        # Position away players
        self._position_players(
            self.away_players, 
            self.away_positions, 
            away_formation, 
            is_home=False
        )
        
        # If no positions were set (no players found), create dummy positions for visualization
        if not self.home_positions:
            self._create_dummy_positions(self.home_lineup, self.home_positions, is_home=True)
            
        if not self.away_positions:
            self._create_dummy_positions(self.away_lineup, self.away_positions, is_home=False)
    
    def _create_dummy_positions(self, lineup, positions_dict, is_home):
        """Create dummy positions for players when no proper formation data is available"""
        print(f"Creating dummy positions for {'home' if is_home else 'away'} team")
        
        # Use a basic 4-4-2 layout
        positions = []
        
        # Goalkeeper
        positions.append((5 if is_home else self.pitch_length - 5, self.pitch_width / 2))
        
        # Defenders - 4
        def_y_start = self.pitch_width * 0.2
        def_y_end = self.pitch_width * 0.8
        def_x = 15 if is_home else self.pitch_length - 15
        for i in range(4):
            y_pos = def_y_start + (def_y_end - def_y_start) * i / 3
            positions.append((def_x, y_pos))
        
        # Midfielders - 4
        mid_y_start = self.pitch_width * 0.15
        mid_y_end = self.pitch_width * 0.85
        mid_x = self.pitch_length * 0.4 if is_home else self.pitch_length * 0.6
        for i in range(4):
            y_pos = mid_y_start + (mid_y_end - mid_y_start) * i / 3
            positions.append((mid_x, y_pos))
        
        # Forwards - 2
        fwd_y_start = self.pitch_width * 0.35
        fwd_y_end = self.pitch_width * 0.65
        fwd_x = self.pitch_length * 0.7 if is_home else self.pitch_length * 0.3
        for i in range(2):
            y_pos = fwd_y_start + (fwd_y_end - fwd_y_start) * i
            positions.append((fwd_x, y_pos))
        
        # Assign positions to players
        for i, (player_id, player) in enumerate(lineup.items()):
            if i < len(positions):
                positions_dict[player_id] = list(positions[i])
    
    def _position_players(self, players, positions_dict, formation, is_home):
        """Position players according to formation"""
        formation_parts = self._parse_formation(formation)
        
        # Goalkeeper - ensure there is a goalkeeper
        if 'GK' in players and players['GK']:
            gk = players['GK'][0]
            if is_home:
                positions_dict[gk['id']] = [5, self.pitch_width / 2]
            else:
                positions_dict[gk['id']] = [self.pitch_length - 5, self.pitch_width / 2]
        else:
            print(f"Warning: No goalkeeper available for {'home' if is_home else 'away'} team.")
        
        # Defenders - check if exists
        if 'DEF' in players and players['DEF']:
            defenders = players['DEF']
            self._position_line(
                defenders, 
                positions_dict, 
                y_start=self.pitch_width * 0.2,
                y_end=self.pitch_width * 0.8,
                x_pos=20 if is_home else self.pitch_length - 20
            )
        else:
            print(f"Warning: No defenders available for {'home' if is_home else 'away'} team.")
        
        # Midfielders - check if exists
        if 'MID' in players and players['MID']:
            midfielders = players['MID']
            self._position_line(
                midfielders, 
                positions_dict, 
                y_start=self.pitch_width * 0.15,
                y_end=self.pitch_width * 0.85,
                x_pos=self.pitch_length * 0.4 if is_home else self.pitch_length * 0.6
            )
        else:
            print(f"Warning: No midfielders available for {'home' if is_home else 'away'} team.")
        
        # Forwards - check if exists
        if 'FWD' in players and players['FWD']:
            forwards = players['FWD']
            self._position_line(
                forwards, 
                positions_dict, 
                y_start=self.pitch_width * 0.25,
                y_end=self.pitch_width * 0.75,
                x_pos=self.pitch_length * 0.7 if is_home else self.pitch_length * 0.3
            )
        else:
            print(f"Warning: No forwards available for {'home' if is_home else 'away'} team.")
    
    def _position_line(self, players, positions_dict, y_start, y_end, x_pos):
        """Position a line of players evenly"""
        if not players:
            return  # No players to position
            
        n_players = len(players)
        
        if n_players == 1:
            # Single player centered
            positions_dict[players[0]['id']] = [x_pos, self.pitch_width / 2]
        else:
            # Multiple players evenly spaced
            for i, player in enumerate(players):
                y_pos = y_start + (y_end - y_start) * i / (n_players - 1)
                positions_dict[player['id']] = [x_pos, y_pos]
    
    def run_simulation(self, duration=90*60, update_callback=None):
        """
        Run the match simulation for the specified duration
        
        Args:
            duration: Duration in seconds (default: 90 minutes)
            update_callback: Optional callback function to call after each event
        """
        while self.current_time < duration and self.event_queue:
            # Get next event from queue
            event_time, event_type, team_id, event_data = self.event_queue.popleft()
            
            # Update simulation time
            self.current_time = event_time
            
            # Process event
            self._process_event(event_type, team_id, event_data)
            
            # Update player positions between events
            self._update_player_positions()
            
            # Schedule next event based on current state if needed
            if not self.event_queue:
                self._predict_next_event()
                
            # Call update callback if provided
            if update_callback:
                update_callback(self)
            
            # Add a small delay for real-time visualization if needed
            # time.sleep(0.05)
    
    def _schedule_event(self, event_type, delay, team_id, event_data=None):
        """
        Schedule an event to happen after the specified delay
        
        Args:
            event_type: Type of event (EventType enum)
            delay: Delay in seconds from current time
            team_id: ID of the team associated with the event
            event_data: Additional data for the event
        """
        if event_data is None:
            event_data = {}
            
        # Add event to queue with appropriate timing
        event_time = self.current_time + delay
        
        # Add to queue and sort by time
        self.event_queue.append((event_time, event_type, team_id, event_data))
        self.event_queue = deque(sorted(self.event_queue, key=lambda x: x[0]))
    
    def _process_event(self, event_type, team_id, event_data):
        """
        Process an event and update the match state
        
        Args:
            event_type: Type of event (EventType enum)
            team_id: ID of the team associated with the event
            event_data: Additional data for the event
        """
        # Create event record
        event_record = {
            'match_id': f"{self.home_team_id}_{self.away_team_id}_{self.current_time}",
            'second': self.current_time,
            'team_id': team_id,
            'player_id': event_data.get('player_id'),
            'event_type': event_type.value,
            'start_x': event_data.get('start_x'),
            'start_y': event_data.get('start_y'),
            'end_x': event_data.get('end_x'),
            'end_y': event_data.get('end_y'),
            'receiver_id': event_data.get('receiver_id'),
            'outcome': event_data.get('outcome')
        }
        
        # Add to events history
        self.events.append(event_record)
        
        # Record current player positions for historical analysis
        positions_record = {
            'time': self.current_time,
            'home_positions': self.home_positions.copy(),
            'away_positions': self.away_positions.copy(),
            'ball_position': self.ball_position.copy(),
            'possession': self.possession
        }
        self.player_positions_history.append(positions_record)
        
        # Handle specific event types
        if event_type == EventType.KICKOFF:
            self._handle_kickoff(team_id, event_data)
        elif event_type == EventType.PASS:
            self._handle_pass(team_id, event_data)
        elif event_type == EventType.CARRY:
            self._handle_carry(team_id, event_data)
        elif event_type == EventType.SHOT:
            self._handle_shot(team_id, event_data)
        elif event_type == EventType.GOAL:
            self._handle_goal(team_id, event_data)
        elif event_type == EventType.BALL_RECOVERY:
            self._handle_ball_recovery(team_id, event_data)
        elif event_type == EventType.DUEL:
            self._handle_duel(team_id, event_data)
        # Add more event handlers as needed
    
    def _handle_kickoff(self, team_id, event_data):
        """Handle kickoff event"""
        # Set possession to kickoff team
        self.possession = team_id
        
        # Place ball at center
        self.ball_position = [self.pitch_length / 2, self.pitch_width / 2]
        
        # Find a central midfielder to carry the ball
        if team_id == self.home_team_id:
            players = self.home_players['MID']
            positions = self.home_positions
        else:
            players = self.away_players['MID']
            positions = self.away_positions
        
        # Find player closest to center
        center = np.array([self.pitch_length / 2, self.pitch_width / 2])
        min_dist = float('inf')
        carrier_id = None
        
        for player in players:
            player_id = player['id']
            pos = np.array(positions[player_id])
            dist = np.linalg.norm(pos - center)
            
            if dist < min_dist:
                min_dist = dist
                carrier_id = player_id
        
        # Update ball carrier
        self.ball_carrier = carrier_id
        
        # Move carrier to ball
        if carrier_id is not None:
            positions[carrier_id] = self.ball_position.copy()
            
        # Schedule first pass after kickoff
        self._schedule_event(
            EventType.PASS,
            delay=2,
            team_id=team_id,
            event_data={
                'player_id': carrier_id,
                'start_x': self.ball_position[0],
                'start_y': self.ball_position[1]
            }
        )
    
    def _handle_pass(self, team_id, event_data):
        """Handle pass event"""
        # Get pass details
        player_id = event_data.get('player_id')
        
        # Determine positions dictionary based on team
        positions = self.home_positions if team_id == self.home_team_id else self.away_positions
        lineup = self.home_lineup if team_id == self.home_team_id else self.away_lineup
        
        # Get or predict starting position
        if 'start_x' in event_data and 'start_y' in event_data:
            start_x = event_data['start_x']
            start_y = event_data['start_y']
        elif player_id in positions:
            start_x, start_y = positions[player_id]
        else:
            # Default to current ball position
            start_x, start_y = self.ball_position
        
        # Update ball and player position
        self.ball_position = [start_x, start_y]
        if player_id in positions:
            positions[player_id] = [start_x, start_y]
        
        # Select or predict receiver
        receiver_id = event_data.get('receiver_id')
        
        if receiver_id is None:
            # Find a valid receiver if none specified
            receiver_id = self._find_pass_receiver(team_id, start_x, start_y)
        
        # Get receiver position
        if receiver_id in positions:
            end_x, end_y = positions[receiver_id]
        else:
            # Random position ahead if no valid receiver
            if team_id == self.home_team_id:
                # Home team attacks left to right
                end_x = min(start_x + 15, self.pitch_length - 5)
            else:
                # Away team attacks right to left
                end_x = max(start_x - 15, 5)
            end_y = start_y + random.uniform(-10, 10)
            end_y = max(5, min(end_y, self.pitch_width - 5))
        
        # Update event data with pass details
        event_data.update({
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'receiver_id': receiver_id
        })
        
        # Determine pass outcome
        # In a full model, this would be predicted based on player skills, distance, etc.
        if receiver_id is not None:
            success_prob = self._calculate_pass_success_probability(
                team_id, player_id, receiver_id, start_x, start_y, end_x, end_y
            )
            success = random.random() < success_prob
        else:
            success = False
            
        # Update event outcome
        event_data['outcome'] = None if success else 'INCOMPLETE'
        
        # Calculate pass duration based on distance
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        pass_duration = 0.5 + distance / 20  # Base time plus distance factor
        
        if success:
            # Pass successful - schedule carry for the receiver
            self._schedule_event(
                EventType.CARRY,
                delay=pass_duration,
                team_id=team_id,
                event_data={
                    'player_id': receiver_id,
                    'start_x': end_x,
                    'start_y': end_y
                }
            )
            
            # Update ball trajectory for visualization
            self.ball_carrier = None  # Ball in the air
        else:
            # Pass failed - schedule ball recovery for the other team
            opponent_team_id = self.away_team_id if team_id == self.home_team_id else self.home_team_id
            
            self._schedule_event(
                EventType.BALL_RECOVERY,
                delay=pass_duration + 1,  # Add extra time for recovery
                team_id=opponent_team_id,
                event_data={
                    'end_x': end_x,
                    'end_y': end_y
                }
            )
            
            # Update possession
            self.possession = opponent_team_id
            self.ball_carrier = None
    
    def _find_pass_receiver(self, team_id, start_x, start_y):
        """Find a suitable pass receiver based on current player positions"""
        # Get positions dictionary based on team
        positions = self.home_positions if team_id == self.home_team_id else self.away_positions
        lineup = self.home_lineup if team_id == self.home_team_id else self.away_lineup
        
        # Find players in good positions
        valid_receivers = []
        
        for player_id, position in positions.items():
            if player_id == self.ball_carrier:
                continue  # Skip the current ball carrier
                
            # Calculate pass difficulty
            dx = position[0] - start_x
            dy = position[1] - start_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Skip if too far
            if distance > 40:
                continue
                
            # Check if home or away team is attacking
            attacking_direction = 1 if team_id == self.home_team_id else -1
            
            # Prefer forward passes in the attacking direction
            forward_factor = dx * attacking_direction
            
            # Calculate receiver score (higher is better)
            # In a real model, this would consider tactical context, player attributes, etc.
            score = forward_factor - distance / 5
            
            valid_receivers.append((player_id, score))
        
        if valid_receivers:
            # Sort by score and pick top receiver
            valid_receivers.sort(key=lambda x: x[1], reverse=True)
            return valid_receivers[0][0]
        else:
            return None
    
    def _calculate_pass_success_probability(self, team_id, passer_id, receiver_id, start_x, start_y, end_x, end_y):
        """Calculate the probability of a successful pass"""
        # Get player data
        lineup = self.home_lineup if team_id == self.home_team_id else self.away_lineup
        
        # Get pass distance
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Base probability decreasing with distance
        base_prob = max(0.2, 1 - distance / 100)
        
        # Adjust based on passer's passing ability
        if passer_id in lineup:
            passer = lineup[passer_id]
            if distance < 15:
                passing_skill = passer.get('short_passing', 50)
            else:
                passing_skill = passer.get('long_passing', 50)
            
            # Scale from 0-100 to a 0.5-1.5 multiplier
            skill_multiplier = 0.5 + passing_skill / 100
            base_prob *= skill_multiplier
        
        # Difficulty modifier based on direction
        # Passes toward opponents' goal are harder to complete, but direction should be based
        # on actual field position, not team identity to avoid home advantage
        attacking_direction = 1 if (start_x < self.pitch_length / 2) else -1
        direction_difficulty = 1 - max(0, (end_x - start_x) * attacking_direction / 50)
        base_prob *= direction_difficulty
        
        # Adjust for interception risk based on opponent pressure
        opponent_team_id = self.away_team_id if team_id == self.home_team_id else self.home_team_id
        opponent_positions = self.away_positions if team_id == self.home_team_id else self.home_positions
        
        # Check for opponents near pass path who could intercept
        for _, pos in opponent_positions.items():
            # Distance from opponent to pass line
            # Using point-line distance formula
            num = abs((end_y - start_y) * pos[0] - (end_x - start_x) * pos[1] + end_x * start_y - end_y * start_x)
            den = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2)
            
            if den > 0:
                dist_to_line = num / den
                
                # Check if opponent is close enough to pass line to intercept
                if dist_to_line < 5:
                    # Calculate how far along the pass the interception would occur
                    t = ((pos[0] - start_x) * (end_x - start_x) + (pos[1] - start_y) * (end_y - start_y)) / (den**2)
                    
                    # Only count interceptions along the pass path (0 <= t <= 1)
                    if 0 <= t <= 1:
                        # Reduce success probability based on proximity
                        base_prob *= (0.7 + 0.3 * min(1, dist_to_line / 5))
        
        # Ensure probability is between 0 and 1
        return max(0.1, min(0.95, base_prob))
    
    def _handle_carry(self, team_id, event_data):
        """Handle carry (dribble) event"""
        player_id = event_data.get('player_id')
        
        # Update ball carrier
        self.ball_carrier = player_id
        
        # Get positions dictionary based on team
        positions = self.home_positions if team_id == self.home_team_id else self.away_positions
        
        # Get starting position
        if 'start_x' in event_data and 'start_y' in event_data:
            start_x = event_data['start_x']
            start_y = event_data['start_y']
        elif player_id in positions:
            start_x, start_y = positions[player_id]
        else:
            # Default to current ball position
            start_x, start_y = self.ball_position
        
        # Update position
        self.ball_position = [start_x, start_y]
        if player_id in positions:
            positions[player_id] = [start_x, start_y]
        
        # Determine carry destination
        # In a real model, this would be predicted based on tactical context
        attacking_direction = 1 if team_id == self.home_team_id else -1
        
        # Default to moving forward in attacking direction
        carry_distance = random.uniform(3, 8)
        end_x = start_x + attacking_direction * carry_distance
        
        # Add some sideways movement
        end_y = start_y + random.uniform(-3, 3)
        
        # Keep within pitch bounds
        end_x = max(2, min(end_x, self.pitch_length - 2))
        end_y = max(2, min(end_y, self.pitch_width - 2))
        
        # Update event data
        event_data.update({
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y
        })
        
        # Calculate duration based on distance and player pace
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Get player attributes
        if player_id in (self.home_lineup if team_id == self.home_team_id else self.away_lineup):
            player = (self.home_lineup if team_id == self.home_team_id else self.away_lineup)[player_id]
            pace = player.get('pace', 50) / 100.0
        else:
            pace = 0.5
            
        carry_duration = distance / (5 + 7 * pace)  # Base speed plus pace bonus
        
        # Check for duels during the carry
        duel_prob = self._check_for_duel(team_id, player_id, start_x, start_y, end_x, end_y)
        
        if random.random() < duel_prob:
            # Schedule a duel
            duel_point_t = random.uniform(0.2, 0.8)  # Where along the path the duel occurs
            duel_x = start_x + duel_point_t * (end_x - start_x)
            duel_y = start_y + duel_point_t * (end_y - start_y)
            
            # Find closest opponent
            opponent_id = self._find_closest_opponent(team_id, duel_x, duel_y)
            
            self._schedule_event(
                EventType.DUEL,
                delay=carry_duration * duel_point_t,
                team_id=team_id,
                event_data={
                    'player_id': player_id,
                    'opponent_id': opponent_id,
                    'start_x': duel_x,
                    'start_y': duel_y
                }
            )
        else:
            # Carry completes - check if in shooting position or continue with next event
            if self._is_shooting_position(team_id, end_x, end_y):
                # Schedule shot
                self._schedule_event(
                    EventType.SHOT,
                    delay=carry_duration + random.uniform(0.2, 1.0),
                    team_id=team_id,
                    event_data={
                        'player_id': player_id,
                        'start_x': end_x,
                        'start_y': end_y
                    }
                )
            else:
                # Schedule pass after carry
                self._schedule_event(
                    EventType.PASS,
                    delay=carry_duration + random.uniform(0.5, 1.5),
                    team_id=team_id,
                    event_data={
                        'player_id': player_id,
                        'start_x': end_x,
                        'start_y': end_y
                    }
                )
    
    def _check_for_duel(self, team_id, player_id, start_x, start_y, end_x, end_y):
        """Check the probability of a duel occurring during a carry"""
        # Get opponent positions
        opponent_positions = self.away_positions if team_id == self.home_team_id else self.home_positions
        
        # Base duel probability
        base_prob = 0.1
        
        # Check for opponents near carrier's path
        for opp_id, pos in opponent_positions.items():
            # Distance from opponent to carrier's path
            num = abs((end_y - start_y) * pos[0] - (end_x - start_x) * pos[1] + end_x * start_y - end_y * start_x)
            den = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2)
            
            if den > 0:
                dist_to_path = num / den
                
                # Check if opponent is close enough to intercept
                if dist_to_path < 5:
                    # Calculate how far along the path the interception would occur
                    t = ((pos[0] - start_x) * (end_x - start_x) + (pos[1] - start_y) * (end_y - start_y)) / (den**2)
                    
                    # Only count interceptions along the path (0 <= t <= 1)
                    if 0 <= t <= 1:
                        # Increase duel probability based on proximity
                        base_prob += 0.3 * (1 - min(1, dist_to_path / 5))
        
        # Adjust based on field position (more duels in midfield)
        field_position_factor = 1 - abs(start_x - self.pitch_length/2) / (self.pitch_length/2)
        base_prob *= (0.7 + 0.6 * field_position_factor)
        
        # Ensure probability is between 0 and 1
        return min(0.8, base_prob)
    
    def _find_closest_opponent(self, team_id, x, y):
        """Find the closest opponent to a given position"""
        opponent_positions = self.away_positions if team_id == self.home_team_id else self.home_positions
        
        min_dist = float('inf')
        closest_id = None
        
        for opp_id, pos in opponent_positions.items():
            dist = np.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_id = opp_id
                
        return closest_id
    
    def _is_shooting_position(self, team_id, x, y):
        """Determine if a position is suitable for shooting"""
        # Get the distance to goal
        if team_id == self.home_team_id:
            # Home team attacks left to right
            goal_x = self.pitch_length
            goal_y = self.pitch_width / 2
        else:
            # Away team attacks right to left
            goal_x = 0
            goal_y = self.pitch_width / 2
            
        # Calculate distance to goal
        dist_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        
        # Calculate angle to goal (center)
        # A narrow angle makes shooting harder
        dy = goal_y - y
        dx = goal_x - x
        angle_to_goal = abs(np.degrees(np.arctan2(dy, dx)))
        
        # More likely to shoot when closer to goal and with better angle
        if dist_to_goal < 20:
            # Inside the box, high shooting probability
            shoot_prob = 0.7 * (1 - angle_to_goal / 90)
        elif dist_to_goal < 30:
            # Outside the box but still in range
            shoot_prob = 0.3 * (1 - angle_to_goal / 90)
        else:
            # Too far for a reasonable shot
            shoot_prob = 0.05
            
        return random.random() < shoot_prob
    
    def _handle_duel(self, team_id, event_data):
        """Handle duel (tackle/challenge) event"""
        player_id = event_data.get('player_id')
        opponent_id = event_data.get('opponent_id')
        
        # Get positions
        if 'start_x' in event_data and 'start_y' in event_data:
            x = event_data['start_x']
            y = event_data['start_y']
        else:
            x, y = self.ball_position
            
        # Update positions to the duel location
        self.ball_position = [x, y]
        
        positions = self.home_positions if team_id == self.home_team_id else self.away_positions
        opponent_positions = self.away_positions if team_id == self.home_team_id else self.home_positions
        
        if player_id in positions:
            positions[player_id] = [x, y]
            
        if opponent_id in opponent_positions:
            opponent_positions[opponent_id] = [x, y]
            
        # Determine duel outcome
        # In a real model, this would use player attributes
        duel_win_prob = 0.5  # Equal chance by default
        
        # Adjust based on player attributes if available
        lineup = self.home_lineup if team_id == self.home_team_id else self.away_lineup
        opponent_lineup = self.away_lineup if team_id == self.home_team_id else self.home_lineup
        
        if player_id in lineup and opponent_id in opponent_lineup:
            player = lineup[player_id]
            opponent = opponent_lineup[opponent_id]
            
            # Use relevant attributes for duels
            player_strength = player.get('strength', 50)
            player_tackling = player.get('tackling', 50)
            player_duel_rating = (player_strength + player_tackling) / 2
            
            opponent_strength = opponent.get('strength', 50)
            opponent_tackling = opponent.get('tackling', 50)
            opponent_duel_rating = (opponent_strength + opponent_tackling) / 2
            
            # Calculate win probability based on relative ratings
            duel_win_prob = player_duel_rating / (player_duel_rating + opponent_duel_rating)
        
        # Determine winner
        duel_won = random.random() < duel_win_prob
        
        # Set delay for next event
        delay = random.uniform(1, 3)
        
        if duel_won:
            # Player wins duel
            self._schedule_event(
                EventType.CARRY,
                delay=delay,
                team_id=team_id,
                event_data={
                    'player_id': player_id,
                    'start_x': x,
                    'start_y': y
                }
            )
            self.ball_carrier = player_id
        else:
            # Opponent wins duel
            opponent_team_id = self.away_team_id if team_id == self.home_team_id else self.home_team_id
            self._schedule_event(
                EventType.BALL_RECOVERY,
                delay=delay,
                team_id=opponent_team_id,
                event_data={
                    'player_id': opponent_id,
                    'start_x': x,
                    'start_y': y
                }
            )
            self.possession = opponent_team_id
            self.ball_carrier = opponent_id
    
    def _handle_shot(self, team_id, event_data):
        """Handle shot event"""
        player_id = event_data.get('player_id')
        
        # Get shot starting position
        if 'start_x' in event_data and 'start_y' in event_data:
            start_x = event_data['start_x']
            start_y = event_data['start_y']
        else:
            start_x, start_y = self.ball_position
        
        # Update positions
        self.ball_position = [start_x, start_y]
        
        positions = self.home_positions if team_id == self.home_team_id else self.away_positions
        if player_id in positions:
            positions[player_id] = [start_x, start_y]
        
        # Determine goal location
        if team_id == self.home_team_id:
            # Home team attacks left to right
            goal_x = self.pitch_length
            goal_y = self.pitch_width / 2
        else:
            # Away team attacks right to left
            goal_x = 0
            goal_y = self.pitch_width / 2
        
        # Calculate shot probability
        shot_success = self._calculate_shot_success(team_id, player_id, start_x, start_y, goal_x, goal_y)
        
        # Update event data
        event_data.update({
            'end_x': goal_x,
            'end_y': goal_y + random.uniform(-3.66/2, 3.66/2),  # Standard goal width is 7.32m (3.66m on each side)
            'outcome': 'GOAL' if shot_success else 'OFF TARGET'
        })
        
        # Calculate shot duration
        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        shot_duration = distance / 30  # Faster than passes
        
        # Ball is in the air during shot
        self.ball_carrier = None
        
        if shot_success:
            # Goal scored
            self._schedule_event(
                EventType.GOAL,
                delay=shot_duration,
                team_id=team_id,
                event_data={
                    'player_id': player_id,
                    'start_x': start_x,
                    'start_y': start_y,
                    'end_x': goal_x,
                    'end_y': event_data['end_y']
                }
            )
        else:
            # Shot missed
            # Schedule ball recovery for the opponent
            opponent_team_id = self.away_team_id if team_id == self.home_team_id else self.home_team_id
            
            # Randomize recovery position
            recovery_x = goal_x + (-1 if team_id == self.home_team_id else 1) * random.uniform(2, 6)
            recovery_y = event_data['end_y'] + random.uniform(-5, 5)
            
            # Keep within pitch bounds
            recovery_x = max(2, min(recovery_x, self.pitch_length - 2))
            recovery_y = max(2, min(recovery_y, self.pitch_width - 2))
            
            self._schedule_event(
                EventType.BALL_RECOVERY,
                delay=shot_duration + random.uniform(1, 3),
                team_id=opponent_team_id,
                event_data={
                    'start_x': recovery_x,
                    'start_y': recovery_y
                }
            )
            
            # Update possession
            self.possession = opponent_team_id
    
    def _calculate_shot_success(self, team_id, player_id, x, y, goal_x, goal_y):
        """Calculate the probability of a shot resulting in a goal"""
        # Calculate distance and angle to goal
        distance = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        
        # Calculate angle to goal (in degrees)
        # A narrow angle makes shooting harder
        dy = goal_y - y
        dx = goal_x - x
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        
        # Base probability decreasing with distance and angle
        base_prob = max(0.05, min(0.75, 1 - distance / 60 - angle / 180))
        
        # Adjust based on shooter's finishing ability
        if player_id:
            lineup = self.home_lineup if team_id == self.home_team_id else self.away_lineup
            if player_id in lineup:
                player = lineup[player_id]
                
                # Use finishing and shot power attributes
                finishing = player.get('finishing', 50)
                shot_power = player.get('shot_power', 50)
                
                # Calculate shot skill as weighted average
                shot_skill = (0.7 * finishing + 0.3 * shot_power) / 100.0
                
                # Apply skill multiplier (0.5 to 1.5 range)
                skill_multiplier = 0.5 + shot_skill
                base_prob *= skill_multiplier
        
        # Get goalkeeper from opposing team
        opponent_team_id = self.away_team_id if team_id == self.home_team_id else self.home_team_id
        opponent_lineup = self.away_lineup if team_id == self.home_team_id else self.home_lineup
        
        # Find goalkeeper
        gk = None
        for player in opponent_lineup.values():
            if player.get('position') == 'GK':
                gk = player
                break
        
        # Adjust for goalkeeper skill
        if gk:
            gk_positioning = gk.get('gk_positioning', 50)
            gk_reflexes = gk.get('reflexes', 50)
            
            # Calculate goalkeeper effectiveness
            gk_skill = (0.6 * gk_positioning + 0.4 * gk_reflexes) / 100.0
            
            # Reduce success probability based on goalkeeper skill (0.5 to 1.0 reduction)
            base_prob *= (1 - 0.5 * gk_skill)
        
        # Ensure probability is between 0 and 1
        return random.random() < max(0.01, min(0.95, base_prob))
    
    def _handle_goal(self, team_id, event_data):
        """Handle goal event"""
        # Update score
        self.score[team_id] += 1
        
        # Find home team name
        home_team_name = "Home Team"
        away_team_name = "Away Team"
        try:
            home_team_row = self.teams_data[self.teams_data['id'] == self.home_team_id]
            if not home_team_row.empty:
                home_team_name = home_team_row.iloc[0]['name']
            
            away_team_row = self.teams_data[self.teams_data['id'] == self.away_team_id]
            if not away_team_row.empty:
                away_team_name = away_team_row.iloc[0]['name']
        except Exception as e:
            print(f"Warning: Could not determine team names: {e}")
        
        # Make sure scores are integers for string formatting
        home_score = int(self.score[self.home_team_id])
        away_score = int(self.score[self.away_team_id])
        
        # Get player ID and format it as a string to avoid format issues
        player_id = str(event_data.get('player_id', 'Unknown'))
        
        # Format time
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        time_str = f"{minutes}:{seconds:02d}"
        
        print(f"GOAL! {home_team_name} {home_score} - {away_score} {away_team_name} (Player {player_id} {time_str})")
        
        # Schedule kickoff for the other team
        opponent_team_id = self.away_team_id if team_id == self.home_team_id else self.home_team_id
        
        self._schedule_event(
            EventType.KICKOFF,
            delay=random.uniform(25, 35),  # Delay for celebration and restart
            team_id=opponent_team_id,
            event_data={}
        )
    
    def _handle_ball_recovery(self, team_id, event_data):
        """Handle ball recovery event"""
        # Find closest player to recovery position
        if 'start_x' in event_data and 'start_y' in event_data:
            x = event_data['start_x']
            y = event_data['start_y']
        else:
            x, y = self.ball_position
            
        # Update ball position
        self.ball_position = [x, y]
        
        # Find nearest player
        positions = self.home_positions if team_id == self.home_team_id else self.away_positions
        lineup = self.home_lineup if team_id == self.home_team_id else self.away_lineup
        
        player_id = event_data.get('player_id')
        
        if player_id is None:
            # Find closest player if not specified
            min_dist = float('inf')
            
            for pid, pos in positions.items():
                dist = np.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    player_id = pid
        
        # Update event data
        event_data['player_id'] = player_id
        
        # Move player to the ball
        if player_id and player_id in positions:
            positions[player_id] = [x, y]
            
        # Update possession and ball carrier
        self.possession = team_id
        self.ball_carrier = player_id
        
        # Schedule next action (carry or pass)
        if random.random() < 0.7:  # 70% chance of carry after recovery
            self._schedule_event(
                EventType.CARRY,
                delay=random.uniform(0.5, 1.5),
                team_id=team_id,
                event_data={
                    'player_id': player_id,
                    'start_x': x,
                    'start_y': y
                }
            )
        else:  # 30% chance of immediate pass
            self._schedule_event(
                EventType.PASS,
                delay=random.uniform(0.5, 1.5),
                team_id=team_id,
                event_data={
                    'player_id': player_id,
                    'start_x': x,
                    'start_y': y
                }
            )
    
    def _update_player_positions(self):
        """Update player positions between events"""
        # In a full implementation, this would use the movement model
        # to predict player movements based on tactical rules
        
        if self.movement_model:
            # Use trained movement model for prediction
            pass
        else:
            # Simple rule-based movement
            self._rule_based_movement()
    
    def _rule_based_movement(self):
        """Simple rule-based player movement"""
        # Move players toward their tactical positions based on ball location
        # and attacking/defending state
        
        # Home team
        self._update_team_positions(
            self.home_positions,
            self.home_lineup,
            attacking=self.possession == self.home_team_id
        )
        
        # Away team
        self._update_team_positions(
            self.away_positions,
            self.away_lineup,
            attacking=self.possession == self.away_team_id
        )
    
    def _update_team_positions(self, positions, lineup, attacking):
        """Update positions for a single team"""
        # Get ball attraction factor - players are drawn to the ball
        ball_attraction = 0.3
        
        # Different movement for attacking vs defending
        if attacking:
            # Players make more forward runs when attacking
            forward_bias = 0.5
            spread_factor = 1.1  # Spread out when attacking
        else:
            # Players drop back when defending
            forward_bias = -0.2
            spread_factor = 0.9  # Compact when defending
            
        # Reference point (shifts team up/down the pitch)
        # When attacking, move reference point forward
        # When defending, move reference point backward
        ref_x = self.ball_position[0] + (10 if attacking else -10)
        ref_y = self.ball_position[1]
        
        # Limit reference point to pitch
        ref_x = max(0, min(ref_x, self.pitch_length))
        ref_y = max(0, min(ref_y, self.pitch_width))
        
        # Update each player's position
        for player_id, current_pos in positions.items():
            # Skip ball carrier (handled elsewhere)
            if player_id == self.ball_carrier:
                continue
                
            # Get player's position
            player = lineup.get(player_id)
            
            if not player:
                continue
                
            # Movement depends on position
            position = player.get('position', '')
            
            # Base target position (varies by position and team state)
            if 'GK' in position:
                # Goalkeeper stays near goal
                if attacking:
                    target_x = 20
                    target_y = self.pitch_width / 2
                else:
                    target_x = 5
                    target_y = self.pitch_width / 2
            elif any(pos in position for pos in ['CB', 'LB', 'RB']):
                # Defenders
                if attacking:
                    target_x = self.pitch_length * 0.65
                    # Spread across width
                    if 'CB' in position:
                        target_y = self.pitch_width / 2
                    elif 'LB' in position:
                        target_y = self.pitch_width * 0.15
                    else:  # RB
                        target_y = self.pitch_width * 0.85
                else:
                    target_x = self.pitch_length * 0.25
                    # Spread across width but narrower
                    if 'CB' in position:
                        target_y = self.pitch_width / 2
                    elif 'LB' in position:
                        target_y = self.pitch_width * 0.25
                    else:  # RB
                        target_y = self.pitch_width * 0.75
            elif any(pos in position for pos in ['CM', 'CDM', 'CAM']):
                # Central midfielders
                if attacking:
                    target_x = self.pitch_length * 0.75
                    target_y = self.pitch_width / 2
                else:
                    target_x = self.pitch_length * 0.4
                    target_y = self.pitch_width / 2
            elif any(pos in position for pos in ['LM', 'LW']):
                # Left-sided midfielders/wingers
                if attacking:
                    target_x = self.pitch_length * 0.8
                    target_y = self.pitch_width * 0.2
                else:
                    target_x = self.pitch_length * 0.45
                    target_y = self.pitch_width * 0.3
            elif any(pos in position for pos in ['RM', 'RW']):
                # Right-sided midfielders/wingers
                if attacking:
                    target_x = self.pitch_length * 0.8
                    target_y = self.pitch_width * 0.8
                else:
                    target_x = self.pitch_length * 0.45
                    target_y = self.pitch_width * 0.7
            elif any(pos in position for pos in ['ST', 'CF']):
                # Strikers/forwards
                if attacking:
                    target_x = self.pitch_length * 0.9
                    target_y = self.pitch_width / 2
                else:
                    target_x = self.pitch_length * 0.6
                    target_y = self.pitch_width / 2
            else:
                # Default
                target_x = self.pitch_length / 2
                target_y = self.pitch_width / 2
            
            # Flip for away team (attacking in opposite direction)
            if not self.home_lineup.get(player_id):
                target_x = self.pitch_length - target_x
            
            # Apply spread factor
            target_y = (target_y - self.pitch_width / 2) * spread_factor + self.pitch_width / 2
            
            # Adjust target based on ball position (attraction to ball)
            target_x = (1 - ball_attraction) * target_x + ball_attraction * self.ball_position[0]
            target_y = (1 - ball_attraction) * target_y + ball_attraction * self.ball_position[1]
            
            # Apply forward/backward bias
            direction = 1 if self.home_lineup.get(player_id) else -1
            target_x += direction * forward_bias * 10
            
            # Keep target within pitch
            target_x = max(2, min(target_x, self.pitch_length - 2))
            target_y = max(2, min(target_y, self.pitch_width - 2))
            
            # Move toward target position (with some randomness)
            move_speed = random.uniform(0.5, 1.5)  # meters per update
            
            # Vector to target
            dx = target_x - current_pos[0]
            dy = target_y - current_pos[1]
            
            # Normalize and scale
            length = max(0.1, np.sqrt(dx*dx + dy*dy))
            dx = dx / length * move_speed
            dy = dy / length * move_speed
            
            # Update position
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            
            # Keep within pitch
            new_x = max(0, min(new_x, self.pitch_length))
            new_y = max(0, min(new_y, self.pitch_width))
            
            # Update position
            positions[player_id] = [new_x, new_y]
    
    def _predict_next_event(self):
        """Predict and schedule the next event based on current state"""
        if self.event_model:
            # Use trained model for prediction
            pass
        else:
            # Simple rule-based event generation
            self._rule_based_next_event()
    
    def _rule_based_next_event(self):
        """Generate the next event based on simple rules"""
        # If no current events, have the team in possession start a new action
        if not self.event_queue:
            # Find player for ball recovery
            self._schedule_event(
                EventType.BALL_RECOVERY,
                delay=1.0,
                team_id=self.possession,
                event_data={
                    'start_x': self.ball_position[0],
                    'start_y': self.ball_position[1]
                }
            )
    
    def visualize_frame(self, ax=None):
        """
        Visualize the current state of the match
        
        Args:
            ax: Optional matplotlib axis to draw on
            
        Returns:
            matplotlib axis
        """
        if ax is None:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
        
        # Clear previous content
        ax.clear()
        
        # Draw pitch
        self._draw_pitch(ax)
        
        # Draw players
        self._draw_players(ax)
        
        # Draw ball
        self._draw_ball(ax)
        
        # Add match information
        self._add_match_info(ax)
        
        return ax
    
    def _draw_pitch(self, ax):
        """Draw the soccer pitch"""
        # Pitch outline
        ax.plot([0, 0, self.pitch_length, self.pitch_length, 0],
                [0, self.pitch_width, self.pitch_width, 0, 0], 'k-', lw=2)
        
        # Halfway line
        ax.plot([self.pitch_length / 2, self.pitch_length / 2],
                [0, self.pitch_width], 'k-', lw=2)
        
        # Center circle
        center_circle = plt.Circle((self.pitch_length / 2, self.pitch_width / 2),
                                   9.15, fill=False, color='k')
        ax.add_patch(center_circle)
        
        # Center dot
        center_dot = plt.Circle((self.pitch_length / 2, self.pitch_width / 2),
                               0.5, color='k')
        ax.add_patch(center_dot)
        
        # Penalty areas
        # Left
        ax.plot([0, 16.5, 16.5, 0],
                [self.pitch_width / 2 - 20.15, self.pitch_width / 2 - 20.15,
                 self.pitch_width / 2 + 20.15, self.pitch_width / 2 + 20.15],
                'k-', lw=2)
        
        # Right
        ax.plot([self.pitch_length, self.pitch_length - 16.5, self.pitch_length - 16.5, self.pitch_length],
                [self.pitch_width / 2 - 20.15, self.pitch_width / 2 - 20.15,
                 self.pitch_width / 2 + 20.15, self.pitch_width / 2 + 20.15],
                'k-', lw=2)
        
        # Goal areas
        # Left
        ax.plot([0, 5.5, 5.5, 0],
                [self.pitch_width / 2 - 9.16, self.pitch_width / 2 - 9.16,
                 self.pitch_width / 2 + 9.16, self.pitch_width / 2 + 9.16],
                'k-', lw=2)
        
        # Right
        ax.plot([self.pitch_length, self.pitch_length - 5.5, self.pitch_length - 5.5, self.pitch_length],
                [self.pitch_width / 2 - 9.16, self.pitch_width / 2 - 9.16,
                 self.pitch_width / 2 + 9.16, self.pitch_width / 2 + 9.16],
                'k-', lw=2)
        
        # Penalty spots
        left_pen = plt.Circle((11, self.pitch_width / 2), 0.5, color='k')
        right_pen = plt.Circle((self.pitch_length - 11, self.pitch_width / 2), 0.5, color='k')
        ax.add_patch(left_pen)
        ax.add_patch(right_pen)
        
        # Goals
        goal_width = 7.32  # meters
        
        # Left goal
        ax.plot([-2, 0], [self.pitch_width / 2 - goal_width / 2, self.pitch_width / 2 - goal_width / 2], 'k-', lw=3)
        ax.plot([-2, 0], [self.pitch_width / 2 + goal_width / 2, self.pitch_width / 2 + goal_width / 2], 'k-', lw=3)
        ax.plot([-2, -2], [self.pitch_width / 2 - goal_width / 2, self.pitch_width / 2 + goal_width / 2], 'k-', lw=3)
        
        # Right goal
        ax.plot([self.pitch_length, self.pitch_length + 2],
                [self.pitch_width / 2 - goal_width / 2, self.pitch_width / 2 - goal_width / 2], 'k-', lw=3)
        ax.plot([self.pitch_length, self.pitch_length + 2],
                [self.pitch_width / 2 + goal_width / 2, self.pitch_width / 2 + goal_width / 2], 'k-', lw=3)
        ax.plot([self.pitch_length + 2, self.pitch_length + 2],
                [self.pitch_width / 2 - goal_width / 2, self.pitch_width / 2 + goal_width / 2], 'k-', lw=3)
        
        # Set bounds
        ax.set_xlim(-5, self.pitch_length + 5)
        ax.set_ylim(-5, self.pitch_width + 5)
        
        # Remove axes
        ax.axis('off')
    
    def _draw_players(self, ax):
        """Draw players on the pitch"""
        # Home team (blue)
        for player_id, pos in self.home_positions.items():
            circle = plt.Circle((pos[0], pos[1]), 1.2, color='blue', alpha=0.7)
            ax.add_patch(circle)
            
            # Add player number if available
            if player_id in self.home_lineup:
                player_name = self.home_lineup[player_id].get('name', str(player_id))
                # Extract last name or short version
                if ' ' in player_name:
                    player_name = player_name.split(' ')[-1]
                player_name = player_name[:5]  # Limit length
                
                ax.text(pos[0], pos[1], player_name, ha='center', va='center', color='white', fontsize=8)
            
            # Highlight if ball carrier
            if player_id == self.ball_carrier and self.possession == self.home_team_id:
                highlight = plt.Circle((pos[0], pos[1]), 1.5, color='blue', fill=False, lw=2)
                ax.add_patch(highlight)
        
        # Away team (red)
        for player_id, pos in self.away_positions.items():
            circle = plt.Circle((pos[0], pos[1]), 1.2, color='red', alpha=0.7)
            ax.add_patch(circle)
            
            # Add player number if available
            if player_id in self.away_lineup:
                player_name = self.away_lineup[player_id].get('name', str(player_id))
                # Extract last name or short version
                if ' ' in player_name:
                    player_name = player_name.split(' ')[-1]
                player_name = player_name[:5]  # Limit length
                
                ax.text(pos[0], pos[1], player_name, ha='center', va='center', color='white', fontsize=8)
            
            # Highlight if ball carrier
            if player_id == self.ball_carrier and self.possession == self.away_team_id:
                highlight = plt.Circle((pos[0], pos[1]), 1.5, color='red', fill=False, lw=2)
                ax.add_patch(highlight)
    
    def _draw_ball(self, ax):
        """Draw the ball on the pitch"""
        ball = plt.Circle((self.ball_position[0], self.ball_position[1]), 0.8, color='white', edgecolor='black')
        ax.add_patch(ball)
    
    def _add_match_info(self, ax):
        """Add match information to the visualization"""
        try:
            # Get team names
            home_name = "Home Team"
            away_name = "Away Team"
            try:
                home_team_row = self.teams_data[self.teams_data['id'] == self.home_team_id]
                if not home_team_row.empty:
                    home_name = home_team_row.iloc[0]['name']
                
                away_team_row = self.teams_data[self.teams_data['id'] == self.away_team_id]
                if not away_team_row.empty:
                    away_name = away_team_row.iloc[0]['name']
            except Exception as e:
                print(f"Warning: Could not determine team names for visualization: {e}")
            
            # Create match info text
            minutes = int(self.current_time // 60)
            seconds = int(self.current_time % 60)
            time_str = f"{minutes}:{seconds:02d}"
            
            # Make sure scores are integers
            home_score = int(self.score[self.home_team_id])
            away_score = int(self.score[self.away_team_id])
            
            score_str = f"{home_name} {home_score} - {away_score} {away_name}"
        except Exception as e:
            print(f"Warning: Error creating match info: {e}")
            time_str = "0:00"
            score_str = "Home 0 - 0 Away"
        
        # Add text to figure
        ax.text(self.pitch_length / 2, -3, score_str, ha='center', fontsize=14, fontweight='bold')
        ax.text(5, -3, time_str, ha='left', fontsize=12)
        
    def create_animation(self, duration=90, fps=10, filename='match_simulation.mp4'):
        """
        Create an animation of the match simulation
        
        Args:
            duration: Duration in seconds to animate
            fps: Frames per second
            filename: Output filename
            
        Returns:
            animation object
        """
        # Store current state
        original_state = {
            'time': self.current_time,
            'score': self.score.copy(),
            'ball_position': self.ball_position.copy(),
            'ball_carrier': self.ball_carrier,
            'possession': self.possession,
            'home_positions': self.home_positions.copy(),
            'away_positions': self.away_positions.copy(),
            'events': self.events.copy(),
            'event_queue': self.event_queue.copy()
        }
        
        # Reset match
        self.reset_match()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Run simulation and store states
        states = []
        
        def update_callback(simulator):
            states.append({
                'time': simulator.current_time,
                'score': simulator.score.copy(),
                'ball_position': simulator.ball_position.copy(),
                'ball_carrier': simulator.ball_carrier,
                'possession': simulator.possession,
                'home_positions': simulator.home_positions.copy(),
                'away_positions': simulator.away_positions.copy()
            })
        
        # Run simulation
        self.run_simulation(duration=duration*60, update_callback=update_callback)
        
        # Animation function
        def animate(i):
            if i < len(states):
                state = states[i]
                
                # Restore state
                self.current_time = state['time']
                self.score = state['score'].copy()
                self.ball_position = state['ball_position'].copy()
                self.ball_carrier = state['ball_carrier']
                self.possession = state['possession']
                self.home_positions = state['home_positions'].copy()
                self.away_positions = state['away_positions'].copy()
                
                # Draw frame
                self.visualize_frame(ax)
            
            return ax
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(states), interval=1000/fps, blit=False
        )
        
        # Save animation
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        
        # Restore original state
        self.current_time = original_state['time']
        self.score = original_state['score'].copy()
        self.ball_position = original_state['ball_position'].copy()
        self.ball_carrier = original_state['ball_carrier']
        self.possession = original_state['possession']
        self.home_positions = original_state['home_positions'].copy()
        self.away_positions = original_state['away_positions'].copy()
        self.events = original_state['events'].copy()
        self.event_queue = original_state['event_queue'].copy()
        
        return anim
    
    def get_events_dataframe(self):
        """
        Get all events as a pandas DataFrame
        
        Returns:
            DataFrame of events
        """
        return pd.DataFrame(self.events)
    
    def get_positions_history_dataframe(self):
        """
        Get player positions history as a pandas DataFrame
        
        Returns:
            DataFrame of player positions over time
        """
        # Create a list to hold flattened records
        flat_records = []
        
        for record in self.player_positions_history:
            time = record['time']
            possession = record['possession']
            ball_x, ball_y = record['ball_position']
            
            # Add home team positions
            for player_id, pos in record['home_positions'].items():
                flat_records.append({
                    'time': time,
                    'team_id': self.home_team_id,
                    'player_id': player_id,
                    'x': pos[0],
                    'y': pos[1],
                    'is_ball_carrier': player_id == self.ball_carrier,
                    'team_in_possession': possession == self.home_team_id,
                    'ball_x': ball_x,
                    'ball_y': ball_y
                })
            
            # Add away team positions
            for player_id, pos in record['away_positions'].items():
                flat_records.append({
                    'time': time,
                    'team_id': self.away_team_id,
                    'player_id': player_id,
                    'x': pos[0],
                    'y': pos[1],
                    'is_ball_carrier': player_id == self.ball_carrier,
                    'team_in_possession': possession == self.away_team_id,
                    'ball_x': ball_x,
                    'ball_y': ball_y
                })
        
        return pd.DataFrame(flat_records)


def run_simulation_example():
    """Run an example simulation using the SoccerMatchSimulator"""
    # Load data
    players_df = pd.read_csv('csv_exports/players.csv')
    teams_df = pd.read_csv('csv_exports/teams.csv')
    team_styles_df = pd.read_csv('csv_exports/team_styles.csv')
    
    # Create simulator
    simulator = SoccerMatchSimulator(
        home_team_id=1,  # Replace with actual team IDs
        away_team_id=2,
        players_data=players_df,
        teams_data=teams_df,
        team_styles_data=team_styles_df
    )
    
    # Run simulation
    simulator.run_simulation(duration=90*60)  # 90 minutes
    
    # Get events data
    events_df = simulator.get_events_dataframe()
    
    # Print summary
    print(f"Match complete: {simulator.score[simulator.home_team_id]} - {simulator.score[simulator.away_team_id]}")
    print(f"Total events: {len(events_df)}")
    
    # Event type distribution
    event_counts = events_df['event_type'].value_counts()
    print("\nEvent distribution:")
    print(event_counts)
    
    # Visualize final state
    plt.figure(figsize=(12, 8))
    simulator.visualize_frame()
    plt.savefig('match_final_state.png')
    plt.close()
    
    # Create animation
    # simulator.create_animation(duration=5, filename='match_excerpt.mp4')
    
    return simulator, events_df


if __name__ == "__main__":
    run_simulation_example()

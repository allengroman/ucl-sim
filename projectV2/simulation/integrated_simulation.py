import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import argparse
from datetime import datetime
import random

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # For access to models

# Import models
from models.offball_movement_predictor import OffBallMovementPredictor
from simple_lem_train import SimpleLEM

# Import data functions
from top_eleven_function import get_top_eleven
from soccer_data_functions import get_team_players, get_top_teams

# Import enhanced movement
from enhanced_movement import EnhancedMovement

class IntegratedMatchSimulator:
    """
    Soccer match simulator that integrates:
    1. LEM model for on-ball event prediction
    2. Off-ball movement model for player positioning
    """
    
    def __init__(self, lem_model, offball_model, 
                 home_team, away_team, 
                 home_players, away_players,
                 home_style, away_style,
                 event_types, field_width=120, field_height=80,
                 duration=90, interval=1, use_enhanced_movement=True):
        """
        Initialize the integrated match simulator.
        
        Args:
            lem_model: Trained LEM model for event prediction
            offball_model: Trained off-ball movement model
            home_team: Home team name/ID
            away_team: Away team name/ID
            home_players: DataFrame of home team players
            away_players: DataFrame of away team players
            home_style: Dictionary with home team style
            away_style: Dictionary with away team style
            event_types: List of possible event types
            field_width: Width of the field in meters
            field_height: Height of the field in meters
            duration: Match duration in minutes
            interval: Simulation interval in seconds
            use_enhanced_movement: Whether to use enhanced movement logic
        """
        self.lem_model = lem_model
        self.offball_model = offball_model
        self.home_team = home_team
        self.away_team = away_team
        
        # Ensure we have only 11 players per team
        self.home_players = home_players.head(11)
        self.away_players = away_players.head(11)
        
        self.home_style = home_style
        self.away_style = away_style
        self.event_types = event_types
        self.field_width = field_width
        self.field_height = field_height
        self.duration = duration
        self.interval = interval
        self.use_enhanced_movement = use_enhanced_movement
        
        # Initialize enhanced movement if selected
        if use_enhanced_movement:
            self.enhanced_movement = EnhancedMovement(field_width=field_width, field_height=field_height)
        
        # Initialize match state
        self.initialize_match()
        
    def initialize_match(self):
        """Initialize the match state"""
        # Current match time in seconds
        self.match_time = 0
        
        # Events history
        self.events = []
        
        # Match statistics
        self.stats = {
            'home_goals': 0,
            'away_goals': 0,
            'home_shots': 0,
            'away_shots': 0,
            'home_possession': 0,
            'away_possession': 0,
            'home_passes': 0,
            'away_passes': 0,
            'home_pass_success': 0,
            'away_pass_success': 0
        }
        
        # Last 5 events for input to LEM
        self.last_events = []
        
        # Current ball holder
        self.ball_team = 'home' if random.random() < 0.5 else 'away'
        self.ball_player_idx = 9  # Likely a forward
        
        # Ball position
        if self.ball_team == 'home':
            self.ball_position = np.array([self.field_width/4, self.field_height/2])
        else:
            self.ball_position = np.array([3*self.field_width/4, self.field_height/2])
        
        # Initialize player positions
        self.initialize_player_positions()
        
        # Player position history for animation
        self.position_history = {
            'time': [0],
            'ball_x': [self.ball_position[0]],
            'ball_y': [self.ball_position[1]]
        }
        
        # Add initial player positions to history
        for i in range(len(self.home_players)):
            self.position_history[f'home_{i}_x'] = [self.home_positions[i][0]]
            self.position_history[f'home_{i}_y'] = [self.home_positions[i][1]]
        
        for i in range(len(self.away_players)):
            self.position_history[f'away_{i}_x'] = [self.away_positions[i][0]]
            self.position_history[f'away_{i}_y'] = [self.away_positions[i][1]]
        
    def initialize_player_positions(self):
        """Initialize player positions based on team formations"""
        # Get formations
        home_formation = self.home_style.get('formation', '4-3-3')
        away_formation = self.away_style.get('formation', '4-3-3')
        
        # Parse formations
        try:
            home_parts = home_formation.split('-')
            home_defenders = int(home_parts[0])
            home_midfielders = int(home_parts[1])
            home_forwards = int(home_parts[2]) if len(home_parts) > 2 else 0
        except:
            home_defenders = 4
            home_midfielders = 4
            home_forwards = 2
            
        try:
            away_parts = away_formation.split('-')
            away_defenders = int(away_parts[0])
            away_midfielders = int(away_parts[1])
            away_forwards = int(away_parts[2]) if len(away_parts) > 2 else 0
        except:
            away_defenders = 4
            away_midfielders = 4
            away_forwards = 2
        
        # Initialize position arrays
        self.home_positions = np.zeros((len(self.home_players), 2))
        self.away_positions = np.zeros((len(self.away_players), 2))
        
        # Position home team (left to right)
        self.position_team(
            self.home_positions, 
            self.home_players, 
            home_defenders, 
            home_midfielders, 
            home_forwards,
            is_home=True
        )
        
        # Position away team (right to left)
        self.position_team(
            self.away_positions, 
            self.away_players, 
            away_defenders, 
            away_midfielders, 
            away_forwards,
            is_home=False
        )
        
    def position_team(self, positions, players, num_defenders, num_midfielders, num_forwards, is_home=True):
        """Position players of a team based on formation"""
        # Ensure we have at most 11 players
        num_players = min(len(positions), 11)
        
        # Field dimensions
        field_half = self.field_width / 2
        
        # Determine side of pitch
        if is_home:
            # Home team - left side
            gk_x = 5
            def_x = field_half * 0.2
            mid_x = field_half * 0.4
            fw_x = field_half * 0.7
        else:
            # Away team - right side
            gk_x = self.field_width - 5
            def_x = self.field_width - field_half * 0.2
            mid_x = self.field_width - field_half * 0.4
            fw_x = self.field_width - field_half * 0.7
        
        # Sort players by position
        gk_idx = []
        def_idx = []
        mid_idx = []
        fw_idx = []
        
        for i, (_, player) in enumerate(players.iterrows()):
            position = player.get('position', 'MF')
            if position == 'GK':
                gk_idx.append(i)
            elif position == 'DF':
                def_idx.append(i)
            elif position in ['MD', 'MF']:
                mid_idx.append(i)
            else:  # FW or any other
                fw_idx.append(i)
        
        # Make sure we have one goalkeeper
        if not gk_idx and num_players > 0:
            gk_idx = [0]
            
        # Distribute remaining players if needed
        if len(def_idx) < num_defenders:
            # Move midfielders to defense
            needed = num_defenders - len(def_idx)
            while needed > 0 and mid_idx:
                def_idx.append(mid_idx.pop(0))
                needed -= 1
        
        if len(mid_idx) < num_midfielders:
            # Move forwards to midfield
            needed = num_midfielders - len(mid_idx)
            while needed > 0 and fw_idx:
                mid_idx.append(fw_idx.pop(0))
                needed -= 1
                
        if len(fw_idx) < num_forwards:
            # Move midfielders to attack
            needed = num_forwards - len(fw_idx)
            while needed > 0 and len(mid_idx) > num_midfielders:
                fw_idx.append(mid_idx.pop(0))
                needed -= 1
        
        # Position goalkeeper
        if gk_idx:
            positions[gk_idx[0]] = [gk_x, self.field_height / 2]
        
        # Position defenders
        for i, idx in enumerate(def_idx[:num_defenders]):
            x = def_x + np.random.normal(0, 2)  # Add some noise
            y = self.field_height / (num_defenders + 1) * (i + 1)
            positions[idx] = [x, y]
            
        # Position midfielders
        for i, idx in enumerate(mid_idx[:num_midfielders]):
            x = mid_x + np.random.normal(0, 3)  # Add some noise
            y = self.field_height / (num_midfielders + 1) * (i + 1)
            positions[idx] = [x, y]
            
        # Position forwards
        for i, idx in enumerate(fw_idx[:num_forwards]):
            x = fw_x + np.random.normal(0, 3)  # Add some noise
            y = self.field_height / (num_forwards + 1) * (i + 1)
            positions[idx] = [x, y]
            
        # Position any remaining players randomly
        positioned = gk_idx + def_idx[:num_defenders] + mid_idx[:num_midfielders] + fw_idx[:num_forwards]
        for i in range(num_players):
            if i not in positioned:
                x = random.uniform(def_x, fw_x)
                y = random.uniform(5, self.field_height-5)
                positions[i] = [x, y]
        
        # Ensure all positions are within bounds
        for i in range(num_players):
            positions[i][0] = np.clip(positions[i][0], 0, self.field_width)
            positions[i][1] = np.clip(positions[i][1], 0, self.field_height)
    
    def get_player_features(self):
        """Extract player features for the off-ball model"""
        # Get number of players
        num_home = len(self.home_players)
        num_away = len(self.away_players)
        total_players = num_home + num_away
        
        # Initialize feature arrays
        player_features = np.zeros((total_players, 20))  # Fixed size for now
        
        # Fill in features for home team
        for i, (_, player) in enumerate(self.home_players.iterrows()):
            player_features[i] = self.extract_player_features(player)
        
        # Fill in features for away team
        for i, (_, player) in enumerate(self.away_players.iterrows()):
            player_features[num_home + i] = self.extract_player_features(player)
        
        return player_features
    
    def extract_player_features(self, player):
        """Extract normalized feature vector for a player"""
        # Initialize with default values
        features = np.ones(20) * 0.5
        
        # Fill in available attributes
        for i, attr in enumerate(['pace', 'stamina', 'strength', 'short_passing', 
                                  'long_passing', 'vision', 'finishing', 'shot_power', 
                                  'dribbling', 'ball_control', 'tackling', 'interceptions', 
                                  'blocking', 'positioning', 'reflexes', 'handling', 
                                  'gk_positioning']):
            if attr in player:
                # Normalize to [0, 1]
                features[i] = player[attr] / 100.0
        
        # One-hot encode position
        pos_encoding = np.zeros(3)
        position = player.get('position', 'MD')
        
        if position == 'GK':
            pos_encoding[0] = 1
        elif position in ['DF', 'CB', 'LB', 'RB']:
            pos_encoding[1] = 1
        elif position in ['MD', 'MF', 'CM', 'LM', 'RM']:
            pos_encoding[2] = 1
        # Forward is all zeros
        
        # Add position encoding to features
        features[17:20] = pos_encoding
        
        return features
    
    def get_team_style_features(self):
        """Get team style features for the off-ball model"""
        # Initialize feature arrays
        team_features = np.zeros((2, 10))  # Fixed size for now
        
        # Fill in home team features
        team_features[0] = self.extract_team_style_features(self.home_style)
        
        # Fill in away team features
        team_features[1] = self.extract_team_style_features(self.away_style)
        
        return team_features
    
    def extract_team_style_features(self, style):
        """Extract team style features"""
        # Initialize with default values
        features = np.ones(10) * 0.5
        
        # Possession style
        if style.get('possession_style') == 'Possession':
            features[0] = 1.0
        elif style.get('possession_style') == 'Defensive':
            features[0] = 0.0
            
        # Defensive style
        if style.get('defensive_style') == 'High Press':
            features[1] = 1.0
        elif style.get('defensive_style') == 'Low-Block':
            features[1] = 0.0
            
        # Defensive line
        if style.get('defensive_line') == 'High':
            features[2] = 1.0
        elif style.get('defensive_line') == 'Low':
            features[2] = 0.0
            
        # Width
        if style.get('width') == 'Wide':
            features[3] = 1.0
        elif style.get('width') == 'Narrow':
            features[3] = 0.0
            
        # Attacking mindset
        if style.get('attacking_mindset') == 'Attacking':
            features[4] = 1.0
        elif style.get('attacking_mindset') == 'Defensive':
            features[4] = 0.0
            
        # Formation encoding
        formation = style.get('formation', '4-3-3')
        try:
            parts = formation.split('-')
            defenders = int(parts[0]) / 5.0  # Normalize 
            midfielders = int(parts[1]) / 5.0
            forwards = int(parts[2]) / 5.0 if len(parts) > 2 else 0.0
            
            features[5] = defenders
            features[6] = midfielders
            features[7] = forwards
        except:
            pass  # Keep defaults
            
        # Team level
        features[8] = style.get('team_level', 50) / 100.0
        
        # Random feature for diversity
        features[9] = random.random()
        
        return features
    
    def get_current_positions(self):
        """Get current positions for all players"""
        # Combine home and away positions
        num_home = len(self.home_positions)
        num_away = len(self.away_positions)
        
        # Create array for all positions
        positions = np.zeros((num_home + num_away, 2))
        
        # Fill in positions
        positions[:num_home] = self.home_positions
        positions[num_home:] = self.away_positions
        
        return positions
    
    def get_ball_possession_mask(self):
        """Get mask indicating which team has ball possession"""
        num_home = len(self.home_positions)
        num_away = len(self.away_positions)
        
        # Create mask
        mask = np.zeros(num_home + num_away, dtype=bool)
        
        # Set based on possession
        if self.ball_team == 'home':
            mask[:num_home] = True
            mask[num_home:] = False
        else:
            mask[:num_home] = False
            mask[num_home:] = True
        
        return mask
    
    def update_player_positions(self):
        """Update player positions using either enhanced movement logic or off-ball movement model"""
        try:
            if self.use_enhanced_movement:
                # Use enhanced movement logic
                self.home_positions, self.away_positions = self.enhanced_movement.update_player_positions(
                    home_players=self.home_players,
                    away_players=self.away_players,
                    home_positions=self.home_positions,
                    away_positions=self.away_positions,
                    ball_position=self.ball_position,
                    ball_team=self.ball_team,
                    home_style=self.home_style,
                    away_style=self.away_style
                )
                
                # Keep ball carrier's position fixed
                if self.ball_team == 'home':
                    self.home_positions[self.ball_player_idx] = self.ball_position
                else:
                    self.away_positions[self.ball_player_idx] = self.ball_position
                    
            elif self.offball_model is not None:
                # Use the neural network off-ball model
                # Prepare input data for the model
                player_features = self.get_player_features()
                team_features = self.get_team_style_features()
                current_positions = self.get_current_positions()
                ball_position = self.ball_position
                possession_mask = self.get_ball_possession_mask()
                
                # Convert to tensors
                player_features_tensor = torch.tensor(player_features, dtype=torch.float32).unsqueeze(0)
                team_features_tensor = torch.tensor(team_features, dtype=torch.float32).unsqueeze(0)
                current_positions_tensor = torch.tensor(current_positions, dtype=torch.float32).unsqueeze(0)
                ball_position_tensor = torch.tensor(ball_position, dtype=torch.float32).unsqueeze(0)
                possession_mask_tensor = torch.tensor(possession_mask, dtype=torch.bool).unsqueeze(0)
                
                # Set model to evaluation mode
                self.offball_model.eval()
                
                # Predict new positions
                with torch.no_grad():
                    new_positions_tensor = self.offball_model(
                        player_features_tensor,
                        team_features_tensor,
                        current_positions_tensor,
                        ball_position_tensor,
                        possession_mask_tensor
                    )
                
                # Convert back to numpy
                new_positions = new_positions_tensor.squeeze(0).numpy()
                
                # Update positions
                num_home = len(self.home_positions)
                self.home_positions = new_positions[:num_home]
                self.away_positions = new_positions[num_home:]
                
                # Keep ball carrier's position fixed
                if self.ball_team == 'home':
                    self.home_positions[self.ball_player_idx] = self.ball_position
                else:
                    self.away_positions[self.ball_player_idx] = self.ball_position
            
            # Ensure all positions are within field bounds
            for i in range(len(self.home_positions)):
                self.home_positions[i][0] = np.clip(self.home_positions[i][0], 0, self.field_width)
                self.home_positions[i][1] = np.clip(self.home_positions[i][1], 0, self.field_height)
                
            for i in range(len(self.away_positions)):
                self.away_positions[i][0] = np.clip(self.away_positions[i][0], 0, self.field_width)
                self.away_positions[i][1] = np.clip(self.away_positions[i][1], 0, self.field_height)
        
        except Exception as e:
            print(f"Error updating player positions: {e}")
    
    def predict_next_event(self):
        """Predict the next match event using the LEM model"""
        # Skip if no LEM model
        if self.lem_model is None:
            # Generate random event
            return self.generate_random_event()
        
        try:
            # TODO: Implement properly using LEM model with context from last events
            # For now, use random event generation
            return self.generate_random_event()
        except Exception as e:
            print(f"Error predicting next event: {e}")
            return self.generate_random_event()
    
    def generate_random_event(self):
        """Generate a random event as fallback"""
        # Select random event type with weighted probabilities
        # Most events in soccer are passes, followed by various other actions
        event_weights = {
            'PASS': 0.70,    # Increased pass frequency
            'DRIBBLE': 0.12, # Slightly reduced dribbles
            'SHOT': 0.06,    # Reduced shots for more realism
            'TACKLE': 0.06,
            'CLEARANCE': 0.05,
            'GOAL': 0.01
        }
        
        # Make sure all event types are included with some weight
        for event in self.event_types:
            if event not in event_weights:
                event_weights[event] = 0.01
                
        # Convert to list for random.choices
        events = list(event_weights.keys())
        weights = list(event_weights.values())
        
        # Select weighted random event
        event_type = random.choices(events, weights=weights, k=1)[0]
        
        # Ball position is start position
        start_x, start_y = self.ball_position
        
        # Generate end position
        if event_type == 'PASS':
            # Pass to random position in forward direction
            if self.ball_team == 'home':
                # Home team moves left to right
                end_x = start_x + random.uniform(5, 30)
            else:
                # Away team moves right to left
                end_x = start_x - random.uniform(5, 30)
                
            # Random vertical movement
            end_y = start_y + random.uniform(-15, 15)
            
            # Ensure within bounds
            end_x = np.clip(end_x, 0, self.field_width)
            end_y = np.clip(end_y, 0, self.field_height)
            
            # Determine outcome (success/failure)
            success = random.random() < 0.7
            
            # Update stats
            if self.ball_team == 'home':
                self.stats['home_passes'] += 1
                if success:
                    self.stats['home_pass_success'] += 1
            else:
                self.stats['away_passes'] += 1
                if success:
                    self.stats['away_pass_success'] += 1
                    
        elif event_type in ['SHOT', 'GOAL']:
            # Shot toward goal
            if self.ball_team == 'home':
                # Home team shoots toward right goal
                end_x = self.field_width
            else:
                # Away team shoots toward left goal
                end_x = 0
                
            # Aim for the center of the goal
            end_y = self.field_height / 2 + random.uniform(-3, 3)
            
            # Check if it's a goal
            if event_type == 'SHOT':
                # Calculate distance to goal
                if self.ball_team == 'home':
                    distance = self.field_width - start_x
                else:
                    distance = start_x
                    
                # Much lower probability for goals
                # In real soccer, conversion rate is around 10-15% for good chances
                # This scales with distance - close shots have better odds
                max_probability = 0.10  # Max probability even for close shots
                goal_probability = max(0, max_probability * (1 - distance/100))
                
                # Adjust based on team quality
                if self.ball_team == 'home':
                    team_bonus = self.home_style.get('team_level', 50) / 100
                else:
                    team_bonus = self.away_style.get('team_level', 50) / 100
                
                # Apply team bonus (better teams score more)
                goal_probability *= (0.5 + 0.5 * team_bonus)
                
                # Random chance of goal
                is_goal = random.random() < goal_probability
                
                if is_goal:
                    event_type = 'GOAL'
                    
                    # Update stats
                    if self.ball_team == 'home':
                        self.stats['home_goals'] += 1
                    else:
                        self.stats['away_goals'] += 1
            else:
                # Update stats if it's already a GOAL event
                if self.ball_team == 'home':
                    self.stats['home_goals'] += 1
                else:
                    self.stats['away_goals'] += 1
            
            # Update shot stats
            if self.ball_team == 'home':
                self.stats['home_shots'] += 1
            else:
                self.stats['away_shots'] += 1
                
        else:
            # Default end position for other events
            end_x = start_x + random.uniform(-5, 5)
            end_y = start_y + random.uniform(-5, 5)
            
            # Ensure within bounds
            end_x = np.clip(end_x, 0, self.field_width)
            end_y = np.clip(end_y, 0, self.field_height)
            
        # Create event dictionary
        event = {
            'match_time': self.match_time,
            'team': self.ball_team,
            'player_idx': self.ball_player_idx,
            'event_type': event_type,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y
        }
        
        return event
    
    def handle_event(self, event):
        """Handle a match event"""
        event_type = event['event_type']
        start_x, start_y = event['start_x'], event['start_y']
        end_x, end_y = event['end_x'], event['end_y']
        
        # Update ball position
        self.ball_position = np.array([end_x, end_y])
        
        # Handle different event types
        if event_type == 'PASS':
            # Find closest player to end position
            if self.ball_team == 'home':
                # Find closest home player
                distances = [np.linalg.norm(pos - self.ball_position) for pos in self.home_positions]
                new_ball_player = np.argmin(distances)
                
                # Check if pass was intercepted
                away_distances = [np.linalg.norm(pos - self.ball_position) for pos in self.away_positions]
                closest_away = np.argmin(away_distances)
                
                if away_distances[closest_away] < distances[new_ball_player] * 0.8:
                    # Intercepted
                    self.ball_team = 'away'
                    self.ball_player_idx = closest_away
                else:
                    # Successful pass
                    self.ball_player_idx = new_ball_player
            else:
                # Find closest away player
                distances = [np.linalg.norm(pos - self.ball_position) for pos in self.away_positions]
                new_ball_player = np.argmin(distances)
                
                # Check if pass was intercepted
                home_distances = [np.linalg.norm(pos - self.ball_position) for pos in self.home_positions]
                closest_home = np.argmin(home_distances)
                
                if home_distances[closest_home] < distances[new_ball_player] * 0.8:
                    # Intercepted
                    self.ball_team = 'home'
                    self.ball_player_idx = closest_home
                else:
                    # Successful pass
                    self.ball_player_idx = new_ball_player
        
        elif event_type == 'SHOT':
            # Shot missed - turnover
            self.ball_team = 'home' if self.ball_team == 'away' else 'away'
            
            # Goalkeeper gets the ball
            self.ball_player_idx = 0  # Assuming goalkeeper is first player
            
            # Place ball at goal area
            if self.ball_team == 'home':
                self.ball_position = np.array([5, self.field_height/2])
            else:
                self.ball_position = np.array([self.field_width-5, self.field_height/2])
        
        elif event_type == 'GOAL':
            # Goal scored - reset positions
            self.ball_team = 'home' if self.ball_team == 'away' else 'away'
            self.ball_player_idx = 9  # Center forward
            
            # Place ball at center
            self.ball_position = np.array([self.field_width/2, self.field_height/2])
            
            # Reset player positions
            self.initialize_player_positions()
        
        # Add event to history
        self.events.append(event)
        
        # Keep only last 5 events for LEM context
        if len(self.last_events) >= 5:
            self.last_events.pop(0)
        self.last_events.append(event)
        
        # Update player carrying the ball
        if self.ball_team == 'home':
            self.home_positions[self.ball_player_idx] = self.ball_position
        else:
            self.away_positions[self.ball_player_idx] = self.ball_position
    
    def step(self):
        """Advance simulation by one step"""
        # Update match time
        self.match_time += self.interval
        
        # Check if match is over
        if self.match_time > self.duration * 60:
            return False
        
        # Predict next event
        next_event = self.predict_next_event()
        
        # Handle the event
        self.handle_event(next_event)
        
        # Update player positions
        self.update_player_positions()
        
        # Update position history
        self.position_history['time'].append(self.match_time)
        self.position_history['ball_x'].append(self.ball_position[0])
        self.position_history['ball_y'].append(self.ball_position[1])
        
        for i in range(len(self.home_positions)):
            self.position_history[f'home_{i}_x'].append(self.home_positions[i][0])
            self.position_history[f'home_{i}_y'].append(self.home_positions[i][1])
        
        for i in range(len(self.away_positions)):
            self.position_history[f'away_{i}_x'].append(self.away_positions[i][0])
            self.position_history[f'away_{i}_y'].append(self.away_positions[i][1])
        
        return True
    
    def run_simulation(self):
        """Run the complete match simulation"""
        print(f"Starting match: {self.home_team} vs {self.away_team}")
        
        # Run until match is over
        while self.step():
            if self.match_time % 60 == 0:  # Print every minute
                print(f"Match time: {int(self.match_time/60)} min - Score: {self.stats['home_goals']} - {self.stats['away_goals']}")
        
        # Print final score
        print(f"\nFinal Score: {self.home_team} {self.stats['home_goals']} - {self.stats['away_goals']} {self.away_team}")
        
        # Print match stats
        print("\nMatch Statistics:")
        print(f"Shots: {self.stats['home_shots']} - {self.stats['away_shots']}")
        home_pass_accuracy = self.stats['home_pass_success'] / max(1, self.stats['home_passes']) * 100
        away_pass_accuracy = self.stats['away_pass_success'] / max(1, self.stats['away_passes']) * 100
        print(f"Passes: {self.stats['home_passes']} - {self.stats['away_passes']}")
        print(f"Pass Accuracy: {home_pass_accuracy:.1f}% - {away_pass_accuracy:.1f}%")
        
        # Return match results
        return {
            'events': self.events,
            'position_history': self.position_history,
            'stats': self.stats
        }
    
    def create_animation(self, save_path=None, interval=100):
        """Create animation of the match"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw pitch
        self.draw_pitch(ax)
        
        # Initialize plots
        ball_plot, = ax.plot([], [], 'ko', markersize=6)
        home_plot, = ax.plot([], [], 'bo', markersize=8)
        away_plot, = ax.plot([], [], 'ro', markersize=8)
        
        # Text for score and time
        score_text = ax.text(self.field_width/2, -5, '', fontsize=12, ha='center')
        
        # Get history length
        history_len = len(self.position_history['time'])
        
        def animate(i):
            # Get data for this frame
            time = self.position_history['time'][i]
            ball_x = self.position_history['ball_x'][i]
            ball_y = self.position_history['ball_y'][i]
            
            # Get home team positions (limit to 11 players)
            max_players = min(len(self.home_players), 11)
            home_x = [self.position_history[f'home_{j}_x'][i] for j in range(max_players)]
            home_y = [self.position_history[f'home_{j}_y'][i] for j in range(max_players)]
            
            # Get away team positions (limit to 11 players)
            max_players = min(len(self.away_players), 11)
            away_x = [self.position_history[f'away_{j}_x'][i] for j in range(max_players)]
            away_y = [self.position_history[f'away_{j}_y'][i] for j in range(max_players)]
            
            # Update plots
            ball_plot.set_data(ball_x, ball_y)
            home_plot.set_data(home_x, home_y)
            away_plot.set_data(away_x, away_y)
            
            # Update score and time
            current_time = int(time / 60)
            current_minute = int(time / 60)
            current_second = int(time % 60)
            
            # Find goals up to this point
            home_goals = 0
            away_goals = 0
            for event in self.events:
                if event['match_time'] <= time and event['event_type'] == 'GOAL':
                    if event['team'] == 'home':
                        home_goals += 1
                    else:
                        away_goals += 1
            
            # Update text
            score_text.set_text(f"{self.home_team} {home_goals} - {away_goals} {self.away_team} ({current_minute:02d}:{current_second:02d})")
            
            return ball_plot, home_plot, away_plot, score_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=history_len, 
            interval=interval, blit=True
        )
        
        # Save if path provided
        if save_path:
            anim.save(save_path, writer='ffmpeg')
            print(f"Animation saved to {save_path}")
        
        return anim
    
    def draw_pitch(self, ax):
        """Draw the soccer pitch"""
        # Field outline
        ax.plot([0, 0, self.field_width, self.field_width, 0], 
                [0, self.field_height, self.field_height, 0, 0], 'k-', lw=2)
        
        # Halfway line
        ax.plot([self.field_width/2, self.field_width/2], 
                [0, self.field_height], 'k-', lw=2)
        
        # Center circle
        center_circle = plt.Circle(
            (self.field_width/2, self.field_height/2), 
            9.15, fill=False, color='k', lw=2
        )
        ax.add_patch(center_circle)
        
        # Penalty areas
        # Home team (left)
        ax.add_patch(Rectangle((0, self.field_height/2 - 20.16), 16.5, 40.32, 
                              fill=False, color='k', lw=2))
        # Away team (right)
        ax.add_patch(Rectangle((self.field_width - 16.5, self.field_height/2 - 20.16), 
                              16.5, 40.32, fill=False, color='k', lw=2))
        
        # Goal areas
        # Home team (left)
        ax.add_patch(Rectangle((0, self.field_height/2 - 9.16), 5.5, 18.32, 
                              fill=False, color='k', lw=2))
        # Away team (right)
        ax.add_patch(Rectangle((self.field_width - 5.5, self.field_height/2 - 9.16), 
                              5.5, 18.32, fill=False, color='k', lw=2))
        
        # Goals
        # Home team (left)
        ax.add_patch(Rectangle((-2, self.field_height/2 - 3.66), 2, 7.32, 
                              fill=False, color='k', lw=2))
        # Away team (right)
        ax.add_patch(Rectangle((self.field_width, self.field_height/2 - 3.66), 
                              2, 7.32, fill=False, color='k', lw=2))
        
        # Set axis limits
        ax.set_xlim(-5, self.field_width + 5)
        ax.set_ylim(-10, self.field_height + 5)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set title
        ax.set_title(f"{self.home_team} vs {self.away_team}")
        
        return ax

def load_lem_model(model_path):
    """Load LEM model from file"""
    try:
        print(f"Loading LEM model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = SimpleLEM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_event_types=checkpoint['num_event_types']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get event types
        event_types = checkpoint.get('encoders', {}).get('event_type', {}).get('categories_', [[]])[0]
        if not event_types:
            event_types = ['PASS', 'SHOT', 'GOAL', 'DRIBBLE', 'TACKLE', 'CLEARANCE']
        
        print(f"LEM model loaded with event types: {event_types}")
        return model, event_types
    except Exception as e:
        print(f"Error loading LEM model: {e}")
        print("Using random event generation instead")
        return None, ['PASS', 'SHOT', 'GOAL', 'DRIBBLE', 'TACKLE', 'CLEARANCE']

def load_offball_model(model_path):
    """Load off-ball movement model from file"""
    try:
        print(f"Loading off-ball movement model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get model parameters
        player_feature_dim = checkpoint.get('player_feature_dim', 20)
        team_style_dim = checkpoint.get('team_style_dim', 10)
        position_dim = checkpoint.get('position_dim', 2)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_heads = checkpoint.get('num_heads', 4)
        num_layers = checkpoint.get('num_layers', 2)
        
        # Create model
        model = OffBallMovementPredictor(
            player_feature_dim=player_feature_dim,
            team_style_dim=team_style_dim,
            position_dim=position_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Off-ball movement model loaded")
        return model
    except Exception as e:
        print(f"Error loading off-ball movement model: {e}")
        print("Using simple physics-based movement instead")
        return None

def load_team_data(team_name, players_file='players.csv', teams_file='teams.csv', team_styles_file='team_styles.csv'):
    """Load team data from CSV files"""
    # Load players
    try:
        players_df = pd.read_csv(players_file)
        team_players = players_df[players_df['team'] == team_name]
        
        if len(team_players) == 0:
            print(f"No players found for team {team_name}. Using random team.")
            team_players = players_df.sample(11)
        
        # Ensure we have exactly 11 players
        if len(team_players) > 11:
            print(f"Limiting {team_name} to 11 players")
            team_players = team_players.head(11)
        elif len(team_players) < 11:
            print(f"Not enough players for {team_name}, adding random players")
            more_players = players_df.sample(11 - len(team_players))
            team_players = pd.concat([team_players, more_players])
    except Exception as e:
        print(f"Error loading players: {e}")
        return None, None
    
    # Load team style
    try:
        teams_df = pd.read_csv(teams_file)
        team_styles_df = pd.read_csv(team_styles_file)
        
        # Get team ID
        team_id = None
        for _, team in teams_df.iterrows():
            if team['name'] == team_name:
                team_id = team['id']
                break
        
        # Get team style
        team_style = None
        if team_id is not None:
            for _, style in team_styles_df.iterrows():
                if 'team_id' in style and style['team_id'] == team_id:
                    team_style = {
                        'possession_style': style.get('possession_style', 'Balanced'),
                        'defensive_style': style.get('defensive_style', 'Mid-Block'),
                        'defensive_line': style.get('defensive_line', 'Medium'),
                        'width': style.get('width', 'Balanced'),
                        'attacking_mindset': style.get('attacking_mindset', 'Balanced'),
                        'formation': style.get('formation', '4-3-3'),
                        'team_level': style.get('team_level', 50)
                    }
                    break
        
        if team_style is None:
            print(f"No style found for team {team_name}. Using default style.")
            team_style = {
                'possession_style': 'Balanced',
                'defensive_style': 'Mid-Block',
                'defensive_line': 'Medium',
                'width': 'Balanced',
                'attacking_mindset': 'Balanced',
                'formation': '4-3-3',
                'team_level': 50
            }
    except Exception as e:
        print(f"Error loading team style: {e}")
        team_style = {
            'possession_style': 'Balanced',
            'defensive_style': 'Mid-Block',
            'defensive_line': 'Medium',
            'width': 'Balanced',
            'attacking_mindset': 'Balanced',
            'formation': '4-3-3',
            'team_level': 50
        }
    
    return team_players, team_style

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run integrated soccer match simulation')
    parser.add_argument('--lem_model', type=str, default='../models/simple_lem_model.pt',
                        help='Path to trained LEM model')
    parser.add_argument('--offball_model', type=str, default='../models/offball_movement_model_final.pt',
                        help='Path to trained off-ball movement model')
    parser.add_argument('--home', type=str, default='Bournemouth',
                        help='Home team name')
    parser.add_argument('--away', type=str, default='Valencia',
                        help='Away team name')
    parser.add_argument('--players', type=str, default='./data/players.csv',
                        help='Path to players CSV file')
    parser.add_argument('--teams', type=str, default='./data/teams.csv',
                        help='Path to teams CSV file')
    parser.add_argument('--team_styles', type=str, default='./data/team_styles.csv',
                        help='Path to team styles CSV file') 
    parser.add_argument('--duration', type=int, default=90,
                        help='Match duration in minutes')
    parser.add_argument('--interval', type=int, default=5,
                        help='Simulation interval in seconds')
    parser.add_argument('--width', type=int, default=120,
                        help='Field width in meters')
    parser.add_argument('--height', type=int, default=80,
                        help='Field height in meters')
    parser.add_argument('--animate', action='store_true',
                        help='Create animation of the match')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results and animations')
    parser.add_argument('--use-enhanced-movement', action='store_true', default=True,
                        help='Use enhanced movement logic instead of neural network model')
    parser.add_argument('--use-neural-movement', action='store_false', dest='use_enhanced_movement',
                        help='Use neural network model instead of enhanced movement logic')
    
    args = parser.parse_args()
    
    # Set current directory to script location for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process team names to handle quoting and spaces 
    home_team = args.home.strip('"\'')  # Remove quotes if present
    away_team = args.away.strip('"\'')  # Remove quotes if present
    
    print(f"Starting simulation: {home_team} vs {away_team}")
    
    # Load models
    lem_model, event_types = load_lem_model(args.lem_model)
    offball_model = load_offball_model(args.offball_model)
    
    # Load team data
    home_players, home_style = load_team_data(
        home_team, 
        players_file=args.players, 
        teams_file=args.teams, 
        team_styles_file=args.team_styles
    )
    
    away_players, away_style = load_team_data(
        away_team, 
        players_file=args.players, 
        teams_file=args.teams, 
        team_styles_file=args.team_styles
    )
    
    # If loading teams failed, use top_eleven to get best players
    if home_players is None or len(home_players) == 0:
        try:
            home_eleven = get_top_eleven(None, formation='4-3-3', players_file=args.players)
            home_players = home_eleven['best_eleven']
        except Exception as e:
            print(f"Error getting top eleven: {e}")
            print("Loading all players and sampling random teams")
            players_df = pd.read_csv(args.players)
            home_players = players_df.sample(11)
    
    if away_players is None or len(away_players) == 0:
        try:
            away_eleven = get_top_eleven(None, formation='4-3-3', players_file=args.players)
            away_players = away_eleven['best_eleven']
        except Exception as e:
            print(f"Error getting top eleven: {e}")
            print("Loading all players and sampling random teams")
            players_df = pd.read_csv(args.players)
            away_players = players_df.sample(11)
    
    # Create simulator
    simulator = IntegratedMatchSimulator(
        lem_model=lem_model,
        offball_model=offball_model,
        home_team=home_team,
        away_team=away_team,
        home_players=home_players,
        away_players=away_players,
        home_style=home_style,
        away_style=away_style,
        event_types=event_types,
        field_width=args.width,
        field_height=args.height,
        duration=args.duration,
        interval=args.interval,
        use_enhanced_movement=args.use_enhanced_movement
    )
    
    # Log which movement system is being used
    if args.use_enhanced_movement:
        print(f"Using enhanced movement logic")
    else:
        print(f"Using neural network movement model")
    
    # Run simulation
    results = simulator.run_simulation()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use clean team names for filenames (replace spaces with underscores)
    home_team_filename = home_team.replace(' ', '_')
    away_team_filename = away_team.replace(' ', '_')
    
    events_file = os.path.join(args.output, f"match_{home_team_filename}_vs_{away_team_filename}_{timestamp}_events.csv")
    positions_file = os.path.join(args.output, f"match_{home_team_filename}_vs_{away_team_filename}_{timestamp}_positions.csv")
    
    # Convert events to DataFrame
    events_df = pd.DataFrame(results['events'])
    events_df.to_csv(events_file, index=False)
    print(f"Events saved to {events_file}")
    
    # Convert position history to DataFrame
    positions_df = pd.DataFrame(results['position_history'])
    positions_df.to_csv(positions_file, index=False)
    print(f"Position history saved to {positions_file}")
    
    # Create animation if requested
    if args.animate:
        animation_file = os.path.join(args.output, f"match_{home_team_filename}_vs_{away_team_filename}_{timestamp}.mp4")
        simulator.create_animation(save_path=animation_file)

if __name__ == "__main__":
    main()
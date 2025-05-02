import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class OffBallMovementPredictor(nn.Module):
    """
    Transformer-based model for predicting off-ball player movement based on 
    game state, team styles, and player attributes.
    """
    def __init__(self, player_feature_dim=20, team_style_dim=10, position_dim=2,
                 hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize the model for predicting off-ball player movements.
        
        Args:
            player_feature_dim: Dimension of player attribute features
            team_style_dim: Dimension of team tactical style features
            position_dim: Dimension of position coordinates (usually 2 for x,y)
            hidden_dim: Size of hidden layers
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super(OffBallMovementPredictor, self).__init__()
        
        # Input embedding layers
        self.player_embedding = nn.Linear(player_feature_dim, hidden_dim)
        self.team_style_embedding = nn.Linear(team_style_dim, hidden_dim)
        self.position_embedding = nn.Linear(position_dim, hidden_dim)
        
        # Transformer encoder for modeling player interactions
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer for position predictions
        self.output_layer = nn.Linear(hidden_dim, position_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, player_features, team_style_features, current_positions, 
                ball_position, possession_team_mask, player_positions_mask=None):
        """
        Forward pass to predict player movements.
        
        Args:
            player_features: Player attribute features [batch_size, num_players, player_feature_dim]
            team_style_features: Team style features [batch_size, num_teams, team_style_dim]
            current_positions: Current player positions [batch_size, num_players, position_dim]
            ball_position: Ball position [batch_size, position_dim]
            possession_team_mask: Binary mask indicating which team has possession [batch_size, num_players]
            player_positions_mask: Attention mask for valid players [batch_size, num_players]
            
        Returns:
            Predicted next positions for all players [batch_size, num_players, position_dim]
        """
        batch_size, num_players, _ = player_features.shape
        
        # Reshape ball position to broadcast to all players
        ball_pos_expanded = ball_position.unsqueeze(1).expand(-1, num_players, -1)
        
        # Compute relative positions to ball
        rel_positions = current_positions - ball_pos_expanded
        
        # Create combined features
        player_embeddings = self.player_embedding(player_features)
        position_embeddings = self.position_embedding(rel_positions)
        
        # Get team embeddings for each player
        # This requires knowing which team each player belongs to
        team_idx = torch.zeros(batch_size, num_players, dtype=torch.long, device=player_features.device)
        team_idx[~possession_team_mask] = 1  # Set index 1 for opposing team
        
        # Select appropriate team style features for each player
        player_team_features = torch.gather(
            team_style_features, 
            1, 
            team_idx.unsqueeze(-1).expand(-1, -1, team_style_features.shape[-1])
        )
        team_embeddings = self.team_style_embedding(player_team_features)
        
        # Combine all embeddings
        combined_embeddings = player_embeddings + position_embeddings + team_embeddings
        combined_embeddings = self.dropout(combined_embeddings)
        
        # Apply transformer to model interactions between players
        if player_positions_mask is not None:
            # Create attention mask
            attn_mask = ~player_positions_mask.unsqueeze(1).expand(-1, num_players, -1)
            transformer_output = self.transformer(combined_embeddings, src_key_padding_mask=attn_mask)
        else:
            transformer_output = self.transformer(combined_embeddings)
        
        # Predict position adjustments
        position_adjustments = self.output_layer(transformer_output)
        
        # Add adjustments to current positions to get new positions
        new_positions = current_positions + position_adjustments
        
        return new_positions

def train_offball_movement_model(events_df, players_df, teams_df, team_styles_df, 
                                 batch_size=32, epochs=20, learning_rate=0.001):
    """
    Train the off-ball movement prediction model.
    
    Args:
        events_df: DataFrame containing match events
        players_df: DataFrame containing player attributes
        teams_df: DataFrame containing team information
        team_styles_df: DataFrame containing team tactical styles
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model
    """
    # Process player data
    player_attributes = process_player_attributes(players_df)
    
    # Process team style data
    team_styles = process_team_styles(team_styles_df)
    
    # Create dataset from events
    dataset = OffBallMovementDataset(
        events_df, 
        player_attributes,
        team_styles,
        teams_df
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = OffBallMovementPredictor(
        player_feature_dim=len(player_attributes[list(player_attributes.keys())[0]]),
        team_style_dim=len(team_styles[list(team_styles.keys())[0]]),
        position_dim=2,
        hidden_dim=128,
        num_heads=4,
        num_layers=2
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function - MSE for position prediction
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in data_loader:
            optimizer.zero_grad()
            
            # Unpack batch
            player_features = batch['player_features']
            team_features = batch['team_features']
            current_positions = batch['current_positions']
            target_positions = batch['target_positions']
            ball_position = batch['ball_position']
            possession_mask = batch['possession_mask']
            
            # Forward pass
            predicted_positions = model(
                player_features,
                team_features,
                current_positions,
                ball_position,
                possession_mask
            )
            
            # Calculate loss
            loss = criterion(predicted_positions, target_positions)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")
    
    return model

def process_player_attributes(players_df):
    """
    Process player attributes into a usable form.
    
    Args:
        players_df: DataFrame containing player data
        
    Returns:
        Dictionary mapping player IDs to attribute vectors
    """
    attribute_cols = [
        'pace', 'stamina', 'strength', 'short_passing', 'long_passing',
        'vision', 'finishing', 'shot_power', 'dribbling', 'ball_control',
        'tackling', 'interceptions', 'blocking', 'positioning', 'reflexes',
        'handling', 'gk_positioning'
    ]
    
    # Normalize attributes to [0, 1]
    players_df_norm = players_df.copy()
    for col in attribute_cols:
        if col in players_df.columns:
            players_df_norm[col] = players_df[col] / 100.0
    
    # Create position encodings
    position_map = {
        'GK': [1, 0, 0],
        'DF': [0, 1, 0],
        'MD': [0, 0, 1],
        'AT': [0, 0, 0]  # Use encoding for other positions
    }
    
    # Create dictionary mapping player IDs to attribute vectors
    player_attributes = {}
    for _, player in players_df_norm.iterrows():
        player_id = player['id']
        
        # Extract position encoding
        position = player.get('position', 'MD')  # Default to midfielder if not specified
        pos_encoding = position_map.get(position, position_map['MD'])
        
        # Create attribute vector
        attributes = []
        for col in attribute_cols:
            if col in player:
                attributes.append(player[col])
            else:
                attributes.append(0.5)  # Default value
        
        # Add position encoding
        attributes.extend(pos_encoding)
        
        # Store in dictionary
        player_attributes[player_id] = np.array(attributes, dtype=np.float32)
    
    return player_attributes

def process_team_styles(team_styles_df):
    """
    Process team tactical styles into feature vectors.
    
    Args:
        team_styles_df: DataFrame containing team style data
        
    Returns:
        Dictionary mapping team IDs to style vectors
    """
    # Tactical feature columns
    tactical_cols = [
        'possession_style', 'defensive_style', 'defensive_line', 
        'width', 'attacking_mindset', 'formation'
    ]
    
    # Statistical feature columns
    stat_cols = [
        'possession_pct', 'ppda', 'progressive_passes', 
        'final_third_entries', 'touches_def_pen', 'touches_att_pen'
    ]
    
    # Create encodings for categorical features
    style_encodings = {
        'possession_style': {
            'Balanced': [0.5, 0.5],
            'Possession': [1.0, 0.0],
            'Defensive': [0.0, 0.5],
            'Counter': [0.0, 1.0]
        },
        'defensive_style': {
            'Mid-Block': [0.5, 0.0],
            'High Press': [1.0, 0.0],
            'Low-Block': [0.0, 0.0]
        },
        'defensive_line': {
            'Low': [0.0],
            'Medium': [0.5],
            'High': [1.0]
        },
        'width': {
            'Balanced': [0.5],
            'Wide': [1.0],
            'Narrow': [0.0]
        },
        'attacking_mindset': {
            'Balanced': [0.5],
            'Attacking': [1.0],
            'Defensive': [0.0]
        }
    }
    
    # Formation encoding function
    def encode_formation(formation):
        # Count defenders, midfielders, attackers
        try:
            nums = formation.split('-')
            # Normalize each position count
            defenders = int(nums[0]) / 5  # Assuming max 5 defenders
            midfielders = int(nums[1]) / 5  # Assuming max 5 midfielders
            
            # Handle formations with different number of parts
            if len(nums) == 3:
                attackers = int(nums[2]) / 5  # Assuming max 5 attackers
            elif len(nums) == 4:
                # For formations like 4-4-1-1
                midfielders += int(nums[2]) / 5
                attackers = int(nums[3]) / 5
            else:
                attackers = 0.0
                
            return [defenders, midfielders, attackers]
        except:
            return [0.4, 0.4, 0.2]  # Default 4-4-2 like formation
    
    # Create dictionary mapping team IDs to style vectors
    team_styles = {}
    for _, style in team_styles_df.iterrows():
        team_id = style['team_id']
        
        # Create style vector
        style_vector = []
        
        # Add tactical encodings
        for col in tactical_cols:
            if col == 'formation':
                style_vector.extend(encode_formation(style[col]))
            elif col in style and col in style_encodings:
                value = style[col]
                encoding = style_encodings[col].get(value, [0.5] * len(next(iter(style_encodings[col].values()))))
                style_vector.extend(encoding)
            else:
                # Default value if missing
                style_vector.extend([0.5] * len(next(iter(style_encodings.get(col, {'default': [0.5]}).values()))))
        
        # Add normalized statistical features
        for col in stat_cols:
            if col in style:
                # Normalize stats
                if col == 'possession_pct':
                    value = style[col] / 100.0
                elif col == 'ppda':
                    value = min(style[col], 100) / 100.0
                elif col == 'progressive_passes':
                    value = min(style[col], 2000) / 2000.0
                elif col == 'final_third_entries':
                    value = min(style[col], 1000) / 1000.0
                elif col.startswith('touches_'):
                    value = min(style[col], 10000) / 10000.0
                else:
                    value = 0.5  # Default
                
                style_vector.append(value)
            else:
                style_vector.append(0.5)  # Default value
        
        # Add team quality (normalized to [0, 1])
        if 'team_level' in style:
            team_quality = style['team_level'] / 100.0
        else:
            team_quality = 0.5  # Default mid-level
        
        style_vector.append(team_quality)
        
        # Store in dictionary
        team_styles[team_id] = np.array(style_vector, dtype=np.float32)
    
    return team_styles

class OffBallMovementDataset(torch.utils.data.Dataset):
    """
    Dataset for training off-ball movement prediction model
    """
    def __init__(self, events_df, player_attributes, team_styles, teams_df):
        """
        Initialize the dataset.
        
        Args:
            events_df: DataFrame containing match events
            player_attributes: Dictionary mapping player IDs to attribute vectors
            team_styles: Dictionary mapping team IDs to style vectors
            teams_df: DataFrame containing team information
        """
        self.events = events_df
        self.player_attributes = player_attributes
        self.team_styles = team_styles
        self.teams_df = teams_df
        
        # Group events by match
        self.match_groups = list(self.events.groupby('match_id'))
        
        # Create indices for accessing sequences
        self.indices = []
        for match_id, match_events in self.match_groups:
            # We need at least 2 consecutive events to predict movement
            if len(match_events) > 3:
                for i in range(len(match_events) - 3):
                    self.indices.append((match_id, i))
    
    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a sequence of events and player positions to predict movement
        
        Returns:
            Dictionary containing features and target positions
        """
        match_id, start_idx = self.indices[idx]
        
        # Find match events
        match_data = None
        for mid, events in self.match_groups:
            if mid == match_id:
                match_data = events
                break
        
        if match_data is None:
            raise ValueError(f"Match ID {match_id} not found")
        
        # Get current event and next event
        current_event = match_data.iloc[start_idx]
        next_event = match_data.iloc[start_idx + 1]
        
        # Extract team IDs
        team1_id = current_event['team_id']
        
        # Find opposing team ID
        team2_id = None
        for _, event in match_data.iterrows():
            if event['team_id'] != team1_id:
                team2_id = event['team_id']
                break
        
        if team2_id is None:
            # If we can't find opposing team, assume next team ID
            team2_id = team1_id + 1
        
        # Get lineup for both teams
        team1_players = self.get_team_players(team1_id)
        team2_players = self.get_team_players(team2_id)
        
        # Combine all players
        all_players = team1_players + team2_players
        num_players = len(all_players)
        
        # Create features tensors
        player_features = np.zeros((num_players, len(next(iter(self.player_attributes.values())))))
        current_positions = np.zeros((num_players, 2))
        target_positions = np.zeros((num_players, 2))
        
        # Fill in player attributes and positions
        for i, player_id in enumerate(all_players):
            # Player attributes
            if player_id in self.player_attributes:
                player_features[i] = self.player_attributes[player_id]
            
            # Current positions - would come from simulation state
            # Here we're generating random positions just for example
            current_positions[i] = self.generate_position(player_id, i < len(team1_players))
            
            # Target positions - would come from next game state
            # Here we're generating random movement for example
            target_positions[i] = self.generate_next_position(current_positions[i])
        
        # Ball position from current event
        if 'start_x' in current_event and 'start_y' in current_event:
            ball_position = np.array([current_event['start_x'], current_event['start_y']])
        else:
            ball_position = np.array([50.0, 40.0])  # Default midfield
        
        # Possession mask - True for players on the team with possession
        possession_team = current_event['team_id']
        possession_mask = np.zeros(num_players, dtype=bool)
        possession_mask[:len(team1_players)] = (possession_team == team1_id)
        possession_mask[len(team1_players):] = (possession_team == team2_id)
        
        # Team features
        team_features = np.zeros((2, len(next(iter(self.team_styles.values())))))
        if team1_id in self.team_styles:
            team_features[0] = self.team_styles[team1_id]
        if team2_id in self.team_styles:
            team_features[1] = self.team_styles[team2_id]
        
        # Convert to tensors
        return {
            'player_features': torch.tensor(player_features, dtype=torch.float32),
            'team_features': torch.tensor(team_features, dtype=torch.float32),
            'current_positions': torch.tensor(current_positions, dtype=torch.float32),
            'target_positions': torch.tensor(target_positions, dtype=torch.float32),
            'ball_position': torch.tensor(ball_position, dtype=torch.float32),
            'possession_mask': torch.tensor(possession_mask)
        }
    
    def get_team_players(self, team_id):
        """Get list of players for a team"""
        # In a real implementation, this would use lineup data
        # Here we're just returning player IDs from the players DataFrame
        team_players = []
        for _, player in self.events.iterrows():
            if 'team_id' in player and player['team_id'] == team_id and 'player_id' in player:
                player_id = player['player_id']
                if player_id not in team_players and player_id in self.player_attributes:
                    team_players.append(player_id)
        
        # Ensure we have 11 players
        while len(team_players) < 11:
            # Generate a placeholder player ID
            placeholder_id = -team_id * 100 - len(team_players)
            team_players.append(placeholder_id)
        
        return team_players[:11]  # Ensure only 11 players
    
    def generate_position(self, player_id, is_team1):
        """Generate a random position for player (for example purposes)"""
        # In a real implementation, this would use actual position data
        # Here we're just generating random positions based on team
        if is_team1:
            x = np.random.uniform(0, 50)
        else:
            x = np.random.uniform(50, 100)
        
        y = np.random.uniform(0, 80)
        return np.array([x, y])
    
    def generate_next_position(self, current_pos):
        """Generate a random next position (for example purposes)"""
        # In a real implementation, this would use actual movement data
        # Here we're just adding random noise to current position
        noise = np.random.normal(0, 3, size=2)
        next_pos = current_pos + noise
        
        # Keep within pitch bounds
        next_pos[0] = np.clip(next_pos[0], 0, 100)
        next_pos[1] = np.clip(next_pos[1], 0, 80)
        
        return next_pos
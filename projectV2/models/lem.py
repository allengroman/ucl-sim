import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict

class TabularLEM(nn.Module):
    """
    Large Events Model (LEM) for soccer match events
    
    This model follows the approach described in the LargeEventsModel repository
    but adapts it to our specific data structure.
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim=256, 
                 num_event_types=10,
                 num_layers=3, 
                 dropout=0.2):
        """
        Initialize the LEM model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Size of hidden layers
            num_event_types: Number of different event types
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super(TabularLEM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_event_types = num_event_types
        self.num_layers = num_layers
        
        # Create the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads for different aspects of an event
        self.event_type_head = nn.Linear(hidden_dim, num_event_types)
        self.spatial_head = nn.Linear(hidden_dim, 4)  # start_x, start_y, end_x, end_y
        self.outcome_head = nn.Linear(hidden_dim, 1)  # success/failure
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with predictions for different aspects of the next event
        """
        # Process sequence through the feature extractor
        features = self.feature_extractor(x)
        
        # Generate predictions
        event_type_logits = F.log_softmax(self.event_type_head(features), dim=1)  # For NLLLoss
        spatial_coords = torch.sigmoid(self.spatial_head(features))  # Normalize to [0,1]
        outcome_prob = torch.sigmoid(self.outcome_head(features))
        
        return {
            'event_type': event_type_logits,
            'spatial': spatial_coords,
            'outcome': outcome_prob
        }
    
    def predict_next_event(self, sequence, temperature=1.0):
        """
        Predict the next event given a sequence
        
        Args:
            sequence: Tensor of recent events (1, seq_len, feature_dim)
            temperature: Temperature for sampling (lower = more deterministic)
            
        Returns:
            Dictionary with sampled predictions for the next event
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            predictions = self(sequence)
            
            # Sample event type with temperature
            event_logits = predictions['event_type'] / max(0.1, temperature)
            event_probs = F.softmax(event_logits, dim=1)
            event_type = torch.multinomial(event_probs, 1).item()
            
            # Get spatial coordinates
            spatial = predictions['spatial'].squeeze(0).numpy()
            
            # Sample outcome
            outcome_prob = predictions['outcome'].item()
            success = np.random.random() < outcome_prob
            
            # Return as dictionary
            return {
                'event_type': event_type,
                'start_x': spatial[0] * 120.0,  # Scale back to original dimensions
                'start_y': spatial[1] * 80.0,
                'end_x': spatial[2] * 120.0,
                'end_y': spatial[3] * 80.0,
                'outcome': None if success else 'INCOMPLETE'
            }

class LEMTrainer:
    """Helper class for training LEM models"""
    def __init__(self, model, device, learning_rate=0.001):
        """
        Initialize the trainer
        
        Args:
            model: LEM model to train
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Define loss functions for different prediction heads
        # Use NLLLoss instead of CrossEntropyLoss to avoid shape issues
        self.loss_functions = {
            'event_type': nn.NLLLoss(),
            'spatial': nn.MSELoss(),
            'outcome': nn.BCELoss()
        }
        
        # Define optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average losses per component
        """
        self.model.train()
        total_losses = defaultdict(float)
        num_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            
            # Extract targets for different components
            event_type_target = targets['event_type'].to(self.device)
            spatial_target = targets['spatial'].to(self.device)
            outcome_target = targets['outcome'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calculate component losses
            # Target is already a long tensor with class indices
            event_type_loss = self.loss_functions['event_type'](
                outputs['event_type'], event_type_target
            )
            
            spatial_loss = self.loss_functions['spatial'](
                outputs['spatial'], spatial_target
            )
            
            outcome_loss = self.loss_functions['outcome'](
                outputs['outcome'].squeeze(1), outcome_target
            )
            
            # Total loss is weighted sum of component losses
            total_loss = event_type_loss + spatial_loss + outcome_loss
            
            # Backward pass and optimize
            total_loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_losses['event_type'] += event_type_loss.item()
            total_losses['spatial'] += spatial_loss.item()
            total_losses['outcome'] += outcome_loss.item()
            total_losses['total'] += total_loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
        
        # Calculate average losses
        for key in total_losses:
            total_losses[key] /= num_batches
            
        return total_losses
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation losses per component
        """
        self.model.eval()
        val_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                
                # Extract targets for different components
                event_type_target = targets['event_type'].to(self.device)
                spatial_target = targets['spatial'].to(self.device)
                outcome_target = targets['outcome'].to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Calculate component losses
                # Target is already a long tensor with class indices
                event_type_loss = self.loss_functions['event_type'](
                    outputs['event_type'], event_type_target
                )
                
                spatial_loss = self.loss_functions['spatial'](
                    outputs['spatial'], spatial_target
                )
                
                outcome_loss = self.loss_functions['outcome'](
                    outputs['outcome'].squeeze(1), outcome_target
                )
                
                # Total loss
                total_loss = event_type_loss + spatial_loss + outcome_loss
                
                # Track losses
                val_losses['event_type'] += event_type_loss.item()
                val_losses['spatial'] += spatial_loss.item()
                val_losses['outcome'] += outcome_loss.item()
                val_losses['total'] += total_loss.item()
                num_batches += 1
        
        # Calculate average losses
        for key in val_losses:
            val_losses[key] /= num_batches
            
        return val_losses

class MatchSimulator:
    """
    Soccer match simulator using LEM for event predictions
    """
    def __init__(self, model, device, encoder_dict):
        """
        Initialize the simulator
        
        Args:
            model: Trained LEM model
            device: Device to run predictions on
            encoder_dict: Dictionary of encoders for categorical variables
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.encoder_dict = encoder_dict
        
        # Set model to eval mode
        self.model.eval()
        
    def simulate_match(self, home_team_id, away_team_id, duration=90*60, temperature=1.0):
        """
        Simulate a complete match
        
        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team
            duration: Match duration in seconds
            temperature: Temperature parameter for sampling
            
        Returns:
            List of events in the match
        """
        # Initialize match state
        current_time = 0
        possession = home_team_id if np.random.random() < 0.5 else away_team_id
        score = {home_team_id: 0, away_team_id: 0}
        events = []
        
        # Create initial seed event (kickoff)
        initial_event = {
            'match_id': f"{home_team_id}_{away_team_id}",
            'second': 0,
            'team_id': possession,
            'player_id': None,  # We don't track specific players in simulation
            'event_type': 'KICKOFF',
            'start_x': 52.5,  # Center of pitch
            'start_y': 34.0,
            'end_x': 52.5,
            'end_y': 34.0,
            'receiver_id': None,
            'outcome': None
        }
        
        events.append(initial_event)
        
        # Initialize event sequence for prediction
        sequence = self._prepare_sequence([initial_event])
        
        # Simulate until match end
        while current_time < duration:
            # Predict next event
            next_event = self.model.predict_next_event(sequence, temperature)
            
            # Advance time
            time_increment = np.random.randint(1, 10)  # Random time between events
            current_time += time_increment
            
            # Create event record
            event_record = {
                'match_id': f"{home_team_id}_{away_team_id}",
                'second': current_time,
                'team_id': possession,
                'player_id': None,  # We don't track specific players
                'event_type': self._decode_event_type(next_event['event_type']),
                'start_x': next_event['start_x'],
                'start_y': next_event['start_y'],
                'end_x': next_event['end_x'],
                'end_y': next_event['end_y'],
                'receiver_id': None,
                'outcome': next_event['outcome']
            }
            
            # Add to events list
            events.append(event_record)
            
            # Update match state based on the event
            self._update_match_state(event_record, score, possession)
            
            # Updates the possession if needed
            if self._is_turnover_event(event_record):
                possession = away_team_id if possession == home_team_id else home_team_id
            
            # Update sequence for next prediction
            sequence = self._update_sequence(sequence, event_record)
            
            # Check for important events (goals, etc.)
            if event_record['event_type'] == 'GOAL':
                score[possession] += 1
                print(f"GOAL! Home {score[home_team_id]} - {score[away_team_id]} Away at {current_time//60}:{current_time%60:02d}")
                
                # Kickoff for other team after goal
                possession = away_team_id if possession == home_team_id else home_team_id
        
        return events
    
    def _prepare_sequence(self, events, max_length=5):
        """
        Prepare a sequence of events for the model
        
        Args:
            events: List of events to convert to tensor
            max_length: Maximum sequence length
            
        Returns:
            Tensor representation of the events
        """
        # Truncate if too long
        if len(events) > max_length:
            events = events[-max_length:]
        
        # Extract features from events
        features = []
        
        for event in events:
            # Create feature vector for an event
            event_features = []
            
            # One-hot encoded event type
            event_type = self.encoder_dict['event_type'].transform([[event['event_type']]])[0]
            event_features.extend(event_type)
            
            # One-hot encoded team
            team_id = self.encoder_dict['team_id'].transform([[event['team_id']]])[0]
            event_features.extend(team_id)
            
            # Spatial features (normalized)
            start_x = event['start_x'] / 120.0 if event['start_x'] is not None else 0.5
            start_y = event['start_y'] / 80.0 if event['start_y'] is not None else 0.5
            end_x = event['end_x'] / 120.0 if event['end_x'] is not None else start_x
            end_y = event['end_y'] / 80.0 if event['end_y'] is not None else start_y
            
            event_features.extend([start_x, start_y, end_x, end_y])
            
            # Temporal feature
            time_feature = event['second'] / (90 * 60.0)  # Normalize to [0,1]
            event_features.append(time_feature)
            
            # Outcome (binary)
            outcome = 1.0 if event['outcome'] is None else 0.0
            event_features.append(outcome)
            
            features.append(event_features)
        
        # Pad sequences if needed
        while len(features) < max_length:
            # Create padding with zeros
            padding = [0.0] * len(features[0])
            features.insert(0, padding)  # Pad at beginning
        
        # Convert to tensor
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def _update_sequence(self, sequence, new_event):
        """
        Update sequence with a new event
        
        Args:
            sequence: Current sequence tensor
            new_event: New event to add
            
        Returns:
            Updated sequence tensor
        """
        # Remove oldest event
        updated_sequence = sequence[:, 1:, :]
        
        # Create feature vector for new event
        new_features = []
        
        # One-hot encoded event type
        event_type = self.encoder_dict['event_type'].transform([[new_event['event_type']]])[0]
        new_features.extend(event_type)
        
        # One-hot encoded team
        team_id = self.encoder_dict['team_id'].transform([[new_event['team_id']]])[0]
        new_features.extend(team_id)
        
        # Spatial features (normalized)
        start_x = new_event['start_x'] / 120.0 if new_event['start_x'] is not None else 0.5
        start_y = new_event['start_y'] / 80.0 if new_event['start_y'] is not None else 0.5
        end_x = new_event['end_x'] / 120.0 if new_event['end_x'] is not None else start_x
        end_y = new_event['end_y'] / 80.0 if new_event['end_y'] is not None else start_y
        
        new_features.extend([start_x, start_y, end_x, end_y])
        
        # Temporal feature
        time_feature = new_event['second'] / (90 * 60.0)  # Normalize to [0,1]
        new_features.append(time_feature)
        
        # Outcome (binary)
        outcome = 1.0 if new_event['outcome'] is None else 0.0
        new_features.append(outcome)
        
        # Add new features to sequence
        new_features_tensor = torch.FloatTensor(new_features).unsqueeze(0).unsqueeze(0).to(self.device)
        return torch.cat([updated_sequence, new_features_tensor], dim=1)
    
    def _decode_event_type(self, event_type_idx):
        """Convert numeric event type to string"""
        # Map index back to event type string
        event_types = self.encoder_dict['event_type'].categories_[0]
        if event_type_idx < len(event_types):
            return event_types[event_type_idx]
        return "PASS"  # Default in case of error
    
    def _update_match_state(self, event, score, possession):
        """Update match state based on event"""
        # Update score on goals
        if event['event_type'] == 'GOAL':
            score[possession] += 1
    
    def _is_turnover_event(self, event):
        """Check if an event results in turnover of possession"""
        # Events that result in possession change
        turnover_events = ['INTERCEPTION', 'BALL RECOVERY', 'GOAL', 'THROW IN', 'OFFSIDE']
        
        # Failed events
        if event['outcome'] == 'INCOMPLETE' and event['event_type'] in ['PASS', 'SHOT']:
            return True
            
        return event['event_type'] in turnover_events
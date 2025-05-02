import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

class SimpleLEM(nn.Module):
    """Simplified Large Events Model"""
    def __init__(self, input_size, hidden_size, num_event_types):
        super(SimpleLEM, self).__init__()
        
        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output heads
        self.event_type_head = nn.Linear(hidden_size, num_event_types)
        self.spatial_head = nn.Linear(hidden_size, 4)  # x, y, end_x, end_y
        self.outcome_head = nn.Linear(hidden_size, 1)  # success probability
    
    def forward(self, x):
        features = self.network(x)
        
        return {
            'event_type': self.event_type_head(features),
            'spatial': torch.sigmoid(self.spatial_head(features)),
            'outcome': torch.sigmoid(self.outcome_head(features))
        }

def load_and_process_data(events_path, num_matches=5):
    """Load and preprocess data"""
    print(f"Loading events from {events_path}")
    df = pd.read_csv(events_path)
    
    # Take a subset for quick training
    match_ids = df['match_id'].unique()[:num_matches]
    print(f"Using {len(match_ids)} matches")
    df = df[df['match_id'].isin(match_ids)]
    
    # Create encoders
    event_encoder = OneHotEncoder(sparse=False)
    event_encoder.fit(df['event_type'].values.reshape(-1, 1))
    
    team_encoder = OneHotEncoder(sparse=False)
    team_encoder.fit(df['team_id'].values.reshape(-1, 1))
    
    # Process sequences
    sequences = []
    targets = []
    seq_length = 5
    
    for match_id in match_ids:
        match_events = df[df['match_id'] == match_id].sort_values('second')
        
        if len(match_events) <= seq_length:
            continue
            
        for i in range(len(match_events) - seq_length):
            # Input sequence
            seq = match_events.iloc[i:i+seq_length]
            
            # Target (next event)
            target = match_events.iloc[i+seq_length]
            
            # Process sequence
            seq_features = []
            for _, event in seq.iterrows():
                # Event type
                event_type = event_encoder.transform([[event['event_type']]])[0]
                
                # Team ID
                team = team_encoder.transform([[event['team_id']]])[0]
                
                # Spatial features
                start_x = event['start_x'] / 120.0 if pd.notna(event['start_x']) else 0.5
                start_y = event['start_y'] / 80.0 if pd.notna(event['start_y']) else 0.5
                end_x = event['end_x'] / 120.0 if pd.notna(event['end_x']) else start_x
                end_y = event['end_y'] / 80.0 if pd.notna(event['end_y']) else start_y
                
                # Time
                time = event['second'] / (90 * 60)
                
                # Outcome
                outcome = 1.0 if pd.isna(event['outcome']) else 0.0
                
                # Combine features
                features = np.concatenate([event_type, team, [start_x, start_y, end_x, end_y, time, outcome]])
                seq_features.append(features)
            
            # Flatten sequence features
            seq_features_flat = np.concatenate(seq_features)
            
            # Process target
            # Event type (as index)
            event_type_idx = np.argmax(event_encoder.transform([[target['event_type']]])[0])
            
            # Spatial features
            target_start_x = target['start_x'] / 120.0 if pd.notna(target['start_x']) else 0.5
            target_start_y = target['start_y'] / 80.0 if pd.notna(target['start_y']) else 0.5
            target_end_x = target['end_x'] / 120.0 if pd.notna(target['end_x']) else target_start_x
            target_end_y = target['end_y'] / 80.0 if pd.notna(target['end_y']) else target_start_y
            
            # Outcome
            target_outcome = 1.0 if pd.isna(target['outcome']) else 0.0
            
            # Add to dataset
            sequences.append(seq_features_flat)
            targets.append({
                'event_type': event_type_idx,
                'spatial': [target_start_x, target_start_y, target_end_x, target_end_y],
                'outcome': target_outcome
            })
    
    # Convert to numpy arrays
    sequences = np.array(sequences, dtype=np.float32)
    
    # Create encoders dict
    encoders = {
        'event_type': event_encoder,
        'team': team_encoder
    }
    
    print(f"Created {len(sequences)} sequences")
    return sequences, targets, encoders

class SoccerDataset(Dataset):
    """Dataset for soccer sequences"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        
        # Create target tensors
        self.event_types = torch.LongTensor([t['event_type'] for t in targets])
        self.spatial = torch.FloatTensor([t['spatial'] for t in targets])
        self.outcomes = torch.FloatTensor([t['outcome'] for t in targets])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'event_type': self.event_types[idx],
            'spatial': self.spatial[idx],
            'outcome': self.outcomes[idx]
        }

def train_model(model, train_loader, val_loader, num_epochs=3, lr=0.001):
    """Train the model"""
    device = torch.device('cpu')
    model.to(device)
    
    # Loss functions
    event_criterion = nn.CrossEntropyLoss()
    spatial_criterion = nn.MSELoss()
    outcome_criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            sequences = batch['sequence'].to(device)
            event_targets = batch['event_type'].to(device)
            spatial_targets = batch['spatial'].to(device)
            outcome_targets = batch['outcome'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            
            # Calculate losses
            event_loss = event_criterion(outputs['event_type'], event_targets)
            spatial_loss = spatial_criterion(outputs['spatial'], spatial_targets)
            # Make sure dimensions match by ensuring outcome_targets has same shape as outputs['outcome']
            outcome_loss = outcome_criterion(outputs['outcome'].view(-1), outcome_targets.view(-1))
            
            # Total loss
            loss = event_loss + spatial_loss + outcome_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                sequences = batch['sequence'].to(device)
                event_targets = batch['event_type'].to(device)
                spatial_targets = batch['spatial'].to(device)
                outcome_targets = batch['outcome'].to(device)
                
                # Forward pass
                outputs = model(sequences)
                
                # Calculate losses
                event_loss = event_criterion(outputs['event_type'], event_targets)
                spatial_loss = spatial_criterion(outputs['spatial'], spatial_targets)
                # Make sure dimensions match by ensuring outcome_targets has same shape as outputs['outcome']
                outcome_loss = outcome_criterion(outputs['outcome'].view(-1), outcome_targets.view(-1))
                
                # Total loss
                loss = event_loss + spatial_loss + outcome_loss
                
                val_loss += loss.item()
        
        # Print metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model

def main():
    """Main function"""
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and process data
    sequences, targets, encoders = load_and_process_data('data/events.csv', num_matches=5)
    
    # Split into train/val
    indices = list(range(len(sequences)))
    random.shuffle(indices)
    
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets
    train_sequences = sequences[train_indices]
    train_targets = [targets[i] for i in train_indices]
    
    val_sequences = sequences[val_indices]
    val_targets = [targets[i] for i in val_indices]
    
    train_dataset = SoccerDataset(train_sequences, train_targets)
    val_dataset = SoccerDataset(val_sequences, val_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
    
    # Get input size from first sequence
    input_size = sequences.shape[1]
    num_event_types = len(encoders['event_type'].categories_[0])
    
    print(f"Input size: {input_size}, Event types: {num_event_types}")
    
    # Create model
    model = SimpleLEM(input_size, hidden_size=128, num_event_types=num_event_types)
    
    # Train model
    model = train_model(model, train_loader, val_loader, num_epochs=3)
    
    # Save model
    model_path = 'models/simple_lem_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': 128,
        'num_event_types': num_event_types,
        'encoders': encoders
    }, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
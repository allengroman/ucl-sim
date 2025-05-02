import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OffBallMovementPredictor model
from models.offball_movement_predictor import OffBallMovementPredictor

class OffBallMovementPredictor(nn.Module):
    """
    Transformer-based model for predicting off-ball player movement based on 
    game state, team styles, and player attributes.
    """
    def __init__(self, player_feature_dim=20, team_style_dim=10, position_dim=2,
                 hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize the model for predicting off-ball player movements.
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
        
        # Get team features for each player
        team_embeddings = torch.zeros_like(player_embeddings)
        for b in range(batch_size):
            for p in range(num_players):
                t_idx = 0 if possession_team_mask[b, p] else 1
                team_embedding = self.team_style_embedding(team_style_features[b, t_idx].unsqueeze(0))
                team_embeddings[b, p] = team_embedding
        
        # Combine all embeddings
        combined_embeddings = player_embeddings + position_embeddings + team_embeddings
        combined_embeddings = self.dropout(combined_embeddings)
        
        # Apply transformer to model interactions between players
        transformer_output = self.transformer(combined_embeddings)
        
        # Predict position adjustments
        position_adjustments = self.output_layer(transformer_output)
        
        # Add adjustments to current positions to get new positions
        new_positions = current_positions + position_adjustments
        
        return new_positions

def generate_synthetic_training_data(num_samples=1000, batch_size=32, num_players=22, seed=42):
    """
    Generate synthetic training data for the off-ball movement model.
    
    Returns:
        DataLoader with batched training samples
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Field dimensions
    field_width = 120
    field_height = 80
    
    # Set up empty lists for batch data
    all_player_features = []
    all_team_features = []
    all_current_positions = []
    all_target_positions = []
    all_ball_positions = []
    all_possession_masks = []
    
    # Generate samples
    for i in range(num_samples):
        # Generate player features - random attributes
        player_features = np.random.rand(num_players, 20)
        
        # Generate team features - tactical styles
        team_features = np.random.rand(2, 10)
        
        # Generate player positions - home team on left side, away team on right side
        home_players = num_players // 2
        current_positions = np.zeros((num_players, 2))
        
        # Position home team players (left side)
        for j in range(home_players):
            if j == 0:  # Goalkeeper
                current_positions[j] = [5, field_height/2]
            elif j <= 4:  # Defenders
                current_positions[j] = [field_width * 0.2, field_height/(5) * j]
            elif j <= 8:  # Midfielders
                current_positions[j] = [field_width * 0.4, field_height/(5) * (j-4)]
            else:  # Forwards
                current_positions[j] = [field_width * 0.7, field_height/(4) * (j-8)]
                
        # Position away team players (right side)
        for j in range(home_players, num_players):
            idx = j - home_players
            if idx == 0:  # Goalkeeper
                current_positions[j] = [field_width - 5, field_height/2]
            elif idx <= 4:  # Defenders
                current_positions[j] = [field_width * 0.8, field_height/(5) * idx]
            elif idx <= 8:  # Midfielders
                current_positions[j] = [field_width * 0.6, field_height/(5) * (idx-4)]
            else:  # Forwards
                current_positions[j] = [field_width * 0.3, field_height/(4) * (idx-8)]
        
        # Add some random variation to positions
        current_positions += np.random.normal(0, 2, current_positions.shape)
        
        # Ensure positions are within field boundaries
        current_positions[:, 0] = np.clip(current_positions[:, 0], 0, field_width)
        current_positions[:, 1] = np.clip(current_positions[:, 1], 0, field_height)
        
        # Generate ball position
        ball_x = np.random.uniform(0, field_width)
        ball_y = np.random.uniform(0, field_height)
        ball_position = np.array([ball_x, ball_y])
        
        # Generate possession mask - one team has possession
        has_possession = np.random.choice([True, False])
        possession_mask = np.zeros(num_players, dtype=bool)
        possession_mask[:home_players] = has_possession
        possession_mask[home_players:] = not has_possession
        
        # Generate target positions - realistic movement toward ball for team with possession
        # and defensive movement for team without possession
        target_positions = current_positions.copy()
        
        # Parameters for movement
        attraction_to_ball = 0.3  # How much players are attracted to the ball
        defensive_movement = 0.2  # How much defensive players move toward their goal
        
        for j in range(num_players):
            # Vector from player to ball
            to_ball = ball_position - current_positions[j]
            dist_to_ball = np.linalg.norm(to_ball)
            
            # Calculate movement
            if dist_to_ball > 0:
                direction_to_ball = to_ball / dist_to_ball
            else:
                direction_to_ball = np.array([0, 0])
                
            movement = np.zeros(2)
            
            # Team with possession moves toward ball
            if possession_mask[j]:
                # Attackers move more toward ball
                if j < home_players and current_positions[j, 0] < ball_position[0]:
                    # Home attacker
                    movement = direction_to_ball * attraction_to_ball * 1.5
                elif j >= home_players and current_positions[j, 0] > ball_position[0]:
                    # Away attacker
                    movement = direction_to_ball * attraction_to_ball * 1.5
                else:
                    # Other players move slightly toward ball
                    movement = direction_to_ball * attraction_to_ball
            else:
                # Team without possession - defensive movement
                if j < home_players:
                    # Home team defends left goal
                    goal_direction = np.array([0, field_height/2]) - current_positions[j]
                else:
                    # Away team defends right goal
                    goal_direction = np.array([field_width, field_height/2]) - current_positions[j]
                    
                # Normalize goal direction
                goal_dist = np.linalg.norm(goal_direction)
                if goal_dist > 0:
                    goal_direction = goal_direction / goal_dist
                
                # Defensive players move more toward goal if ball is close
                if (j < home_players and ball_position[0] < field_width * 0.3) or \
                   (j >= home_players and ball_position[0] > field_width * 0.7):
                    # Ball is close to goal - defend more
                    movement = goal_direction * defensive_movement * 2
                else:
                    # Ball is far from goal - move toward ball slightly
                    movement = direction_to_ball * attraction_to_ball * 0.5
            
            # Add some random noise
            movement += np.random.normal(0, 0.5, 2)
            
            # Update position
            target_positions[j] = current_positions[j] + movement
            
        # Ensure target positions are within field boundaries
        target_positions[:, 0] = np.clip(target_positions[:, 0], 0, field_width)
        target_positions[:, 1] = np.clip(target_positions[:, 1], 0, field_height)
        
        # Add to batch lists
        all_player_features.append(player_features)
        all_team_features.append(team_features)
        all_current_positions.append(current_positions)
        all_target_positions.append(target_positions)
        all_ball_positions.append(ball_position)
        all_possession_masks.append(possession_mask)
    
    # Convert to numpy arrays
    all_player_features = np.array(all_player_features)
    all_team_features = np.array(all_team_features)
    all_current_positions = np.array(all_current_positions)
    all_target_positions = np.array(all_target_positions)
    all_ball_positions = np.array(all_ball_positions)
    all_possession_masks = np.array(all_possession_masks)
    
    # Convert to PyTorch tensors
    tensor_player_features = torch.tensor(all_player_features, dtype=torch.float32)
    tensor_team_features = torch.tensor(all_team_features, dtype=torch.float32)
    tensor_current_positions = torch.tensor(all_current_positions, dtype=torch.float32)
    tensor_target_positions = torch.tensor(all_target_positions, dtype=torch.float32)
    tensor_ball_positions = torch.tensor(all_ball_positions, dtype=torch.float32)
    tensor_possession_masks = torch.tensor(all_possession_masks, dtype=torch.bool)
    
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(
        tensor_player_features,
        tensor_team_features,
        tensor_current_positions,
        tensor_target_positions,
        tensor_ball_positions,
        tensor_possession_masks
    )
    
    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return data_loader

def train_offball_model(data_loader, epochs=10, lr=0.001, model_save_path=None):
    """
    Train the off-ball movement model.
    
    Args:
        data_loader: DataLoader with training data
        epochs: Number of training epochs
        lr: Learning rate
        model_save_path: Path to save the trained model
        
    Returns:
        Trained model and training losses
    """
    # Create model
    model = OffBallMovementPredictor(
        player_feature_dim=20,
        team_style_dim=10,
        position_dim=2,
        hidden_dim=128,
        num_heads=4,
        num_layers=2
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (player_features, team_features, current_positions, 
                        target_positions, ball_positions, possession_masks) in enumerate(data_loader):
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_positions = model(
                player_features,
                team_features,
                current_positions,
                ball_positions,
                possession_masks
            )
            
            # Compute loss
            loss = criterion(predicted_positions, target_positions)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.6f}")
        
        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.6f}")
    
    # Save the trained model
    if model_save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = model_save_path.replace('.pt', f'_epoch{epochs}_{timestamp}.pt')
        
        # Save model and metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'epochs': epochs,
            'loss': losses,
            'player_feature_dim': 20,
            'team_style_dim': 10,
            'position_dim': 2,
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'timestamp': timestamp
        }, save_path)
        
        # Also save as 'final' model
        final_path = model_save_path.replace('.pt', f'_final_{timestamp}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epochs': epochs,
            'loss': losses,
            'player_feature_dim': 20,
            'team_style_dim': 10,
            'position_dim': 2,
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'timestamp': timestamp
        }, final_path)
        
        print(f"Model saved to {save_path} and {final_path}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Save plot if model was saved
    if model_save_path:
        plot_path = os.path.join(
            os.path.dirname(model_save_path),
            f'loss_curve_{timestamp}.png'
        )
        plt.savefig(plot_path)
        print(f"Loss curve saved to {plot_path}")
    
    plt.close()
    
    return model, losses

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train off-ball movement model with synthetic data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='../models/offball_movement_model.pt',
                        help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Set current directory to script location for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Generating {args.samples} synthetic training samples...")
    data_loader = generate_synthetic_training_data(
        num_samples=args.samples,
        batch_size=args.batch_size
    )
    
    print(f"Training the model with {args.epochs} epochs and batch size {args.batch_size}...")
    model, losses = train_offball_model(
        data_loader,
        epochs=args.epochs,
        lr=args.lr,
        model_save_path=args.model_path
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import random

from simple_lem_train import SimpleLEM

def load_model(model_path):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
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
    event_types = checkpoint['encoders']['event_type'].categories_[0]
    
    return model, event_types

def generate_random_events(num_events, event_types, home_team_id, away_team_id):
    """Generate a series of random events based on the model's trained event types"""
    events = []
    current_time = 0
    
    for i in range(num_events):
        # Random time increment
        time_increment = random.randint(1, 10)
        current_time += time_increment
        
        # Random team
        team_id = home_team_id if random.random() < 0.5 else away_team_id
        
        # Random event type
        event_type = random.choice(event_types)
        
        # Random positions
        start_x = random.uniform(0, 120)
        start_y = random.uniform(0, 80)
        end_x = random.uniform(0, 120)
        end_y = random.uniform(0, 80)
        
        # More likely to score when close to goal
        is_goal = False
        if event_type == 'SHOT':
            if team_id == home_team_id:
                # Home team shoots toward right goal
                distance_to_goal = np.sqrt((120 - start_x)**2 + (40 - start_y)**2)
                if distance_to_goal < 25 and random.random() < 0.4:
                    event_type = 'GOAL'
                    is_goal = True
            else:
                # Away team shoots toward left goal
                distance_to_goal = np.sqrt(start_x**2 + (40 - start_y)**2)
                if distance_to_goal < 25 and random.random() < 0.4:
                    event_type = 'GOAL'
                    is_goal = True
        
        # Create event
        event = {
            'match_id': f"{home_team_id}_{away_team_id}",
            'second': current_time,
            'team_id': team_id,
            'player_id': None,
            'event_type': event_type,
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'receiver_id': None,
            'outcome': None if random.random() < 0.7 else 'INCOMPLETE'
        }
        
        events.append(event)
    
    return pd.DataFrame(events)

def visualize_match(events_df, home_team_id, away_team_id, save_path=None):
    """Visualize the match events"""
    plt.figure(figsize=(12, 8))
    
    # Setup pitch
    pitch_length = 120
    pitch_width = 80
    
    # Draw pitch outline
    plt.plot([0, 0, pitch_length, pitch_length, 0], 
             [0, pitch_width, pitch_width, 0, 0], 'k-', lw=2)
    
    # Draw halfway line
    plt.plot([pitch_length/2, pitch_length/2], [0, pitch_width], 'k-', lw=2)
    
    # Draw center circle
    center_circle = plt.Circle((pitch_length/2, pitch_width/2), 9.15, 
                              fill=False, color='k', lw=2)
    plt.gca().add_patch(center_circle)
    
    # Plot events by team
    for team_id, color, label in [(home_team_id, 'blue', 'Home Team'), 
                                 (away_team_id, 'red', 'Away Team')]:
        team_events = events_df[events_df['team_id'] == team_id]
        
        # Plot event positions
        plt.scatter(team_events['start_x'], team_events['start_y'], 
                   color=color, alpha=0.6, label=label)
        
        # Plot passes
        passes = team_events[team_events['event_type'] == 'PASS']
        for _, event in passes.iterrows():
            if pd.notna(event['end_x']) and pd.notna(event['end_y']):
                plt.arrow(event['start_x'], event['start_y'], 
                         event['end_x'] - event['start_x'], 
                         event['end_y'] - event['start_y'],
                         head_width=2, head_length=2, fc=color, ec=color, alpha=0.3)
    
    # Plot goals
    goals = events_df[events_df['event_type'] == 'GOAL']
    plt.scatter(goals['start_x'], goals['start_y'], color='gold', 
               marker='*', s=200, label='Goal')
    
    # Add title and legend
    plt.title(f'Match Simulation: Team {home_team_id} vs Team {away_team_id}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Set axis properties
    plt.xlim(-5, pitch_length+5)
    plt.ylim(-5, pitch_width+5)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main(args):
    """Main function"""
    # Load model (just to get event types, we won't use it for prediction)
    model, event_types = load_model(args.model)
    
    # Generate random events using trained event types
    events_df = generate_random_events(
        num_events=args.duration * 6,  # ~6 events per minute
        event_types=event_types,
        home_team_id=args.home,
        away_team_id=args.away
    )
    
    # Count goals
    home_goals = sum((events_df['team_id'] == args.home) & (events_df['event_type'] == 'GOAL'))
    away_goals = sum((events_df['team_id'] == args.away) & (events_df['event_type'] == 'GOAL'))
    
    print(f"Final Score: {args.home} {home_goals} - {away_goals} {args.away}")
    print(f"Total events: {len(events_df)}")
    
    # Show event distribution
    event_counts = events_df['event_type'].value_counts()
    print("\nEvent distribution:")
    print(event_counts)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = f"{results_dir}/basic_lem_{args.home}_vs_{args.away}_{timestamp}.csv"
    events_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    # Visualize if requested
    if args.visualize:
        viz_path = f"{results_dir}/basic_viz_{args.home}_vs_{args.away}_{timestamp}.png"
        visualize_match(
            events_df=events_df,
            home_team_id=args.home,
            away_team_id=args.away,
            save_path=viz_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run basic soccer match simulation')
    parser.add_argument('--model', type=str, default='models/simple_lem_model.pt',
                        help='Path to trained LEM model')
    parser.add_argument('--home', type=int, default=57,
                        help='Home team ID')
    parser.add_argument('--away', type=int, default=39,
                        help='Away team ID')
    parser.add_argument('--duration', type=int, default=90,
                        help='Match duration in minutes')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize match events')
    
    args = parser.parse_args()
    main(args)
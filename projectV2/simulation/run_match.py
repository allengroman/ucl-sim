#!/usr/bin/env python3
"""
Simple launcher for soccer match simulations
"""
import os
import subprocess
import argparse
import glob

def list_teams():
    """List available teams"""
    try:
        import pandas as pd
        # Try to find the players.csv file
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        players_path = os.path.join(parent_dir, 'data', 'players.csv')
        
        if os.path.exists(players_path):
            # Load the players CSV
            players_df = pd.read_csv(players_path)
            # Get unique teams
            teams = sorted(players_df['team'].unique())
            print("\nAvailable teams:")
            for i, team in enumerate(teams):
                print(f"{i+1}. {team}")
            print("\n")
        else:
            print(f"Could not find players.csv at {players_path}")
    except Exception as e:
        print(f"Error listing teams: {e}")

def find_latest_model(model_type):
    """Find the latest model of the specified type"""
    try:
        # Parent directory
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look in the models directory
        models_dir = os.path.join(parent_dir, '..', 'models')
        
        if model_type == 'lem':
            pattern = os.path.join(models_dir, 'simple_lem_model*.pt')
        else:  # offball
            pattern = os.path.join(models_dir, 'offball_movement_model_final*.pt')
        
        # Find all matching models
        models = glob.glob(pattern)
        
        # Sort by modification time (newest first)
        models.sort(key=os.path.getmtime, reverse=True)
        
        if models:
            return models[0]
        else:
            return None
    except Exception as e:
        print(f"Error finding models: {e}")
        return None

def main():
    """Main function"""
    print("\n===== Soccer Match Simulator =====\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run a soccer match simulation')
    parser.add_argument('--home', type=str, help='Home team name')
    parser.add_argument('--away', type=str, help='Away team name')
    parser.add_argument('--list-teams', action='store_true', help='List available teams')
    parser.add_argument('--animate', action='store_true', help='Create match animation')
    parser.add_argument('--duration', type=int, default=90, help='Match duration in minutes')
    parser.add_argument('--silent', action='store_true', help='Run silently without showing teams')
    
    args = parser.parse_args()
    
    # If listing teams, show teams and exit
    if args.list_teams:
        list_teams()
        return
    
    # If teams not provided, ask for input
    home_team = args.home
    away_team = args.away
    
    if not args.silent:
        list_teams()
    
    if not home_team:
        home_team = input("Enter home team name: ")
    
    if not away_team:
        away_team = input("Enter away team name: ")
    
    # Find latest models
    lem_model = find_latest_model('lem')
    offball_model = find_latest_model('offball')
    
    # Prepare command
    cmd = ['python', 'integrated_simulation.py', 
           '--home', home_team, 
           '--away', away_team,
           '--duration', str(args.duration)]
    
    if args.animate:
        cmd.append('--animate')
    
    if lem_model:
        cmd.extend(['--lem_model', lem_model])
    
    if offball_model:
        cmd.extend(['--offball_model', offball_model])
    
    # Run simulation
    print(f"\nRunning match: {home_team} vs {away_team}")
    if args.animate:
        print("Generating match animation (this may take a while)...")
    
    subprocess.run(cmd)
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
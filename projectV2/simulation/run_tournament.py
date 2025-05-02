#!/usr/bin/env python3
"""
Champions League Tournament Launcher

This script provides a simple way to run the Champions League tournament simulation.
"""
import os
import sys
import argparse
import subprocess
import time

def run_tournament(args):
    """Run the Champions League tournament simulation"""
    print("\n🏆 Champions League Tournament Simulator 🏆\n")
    
    # Build command for tournament simulation
    cmd = ["python", "champions_league.py"]
    
    # Add arguments
    if args.animate:
        cmd.append("--animate")
    
    if args.group_duration:
        cmd.extend(["--group-duration", str(args.group_duration)])
    
    if args.knockout_duration:
        cmd.extend(["--knockout-duration", str(args.knockout_duration)])
        
    # Add movement model selection
    if args.use_enhanced_movement:
        cmd.append("--use-enhanced-movement")
    else:
        cmd.append("--use-neural-movement")
    
    # Run tournament simulation
    print("Starting tournament simulation...")
    print("This may take a while, especially if animations are enabled.")
    
    start_time = time.time()
    
    try:
        process = subprocess.run(cmd, check=True)
        
        # If tournament completed successfully, get the output directory
        # The directory name is 'tournament_results/tournament_TIMESTAMP'
        # We need to find the most recent tournament directory
        tournament_base_dir = "tournament_results"
        if os.path.exists(tournament_base_dir):
            tournament_dirs = [os.path.join(tournament_base_dir, d) for d in os.listdir(tournament_base_dir)
                              if d.startswith("tournament_")]
            if tournament_dirs:
                # Sort by modification time (newest first)
                tournament_dirs.sort(key=os.path.getmtime, reverse=True)
                tournament_dir = tournament_dirs[0]
                
                # Generate visualizations
                if args.visualize:
                    print("\nGenerating visualizations...")
                    viz_cmd = ["python", "visualize_tournament.py", 
                              "--tournament-dir", tournament_dir]
                    subprocess.run(viz_cmd, check=True)
                
                # Print final message
                print(f"\nTournament completed successfully!")
                print(f"Results saved to: {tournament_dir}")
                print(f"Time taken: {(time.time() - start_time) / 60:.1f} minutes")
                
                return tournament_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tournament: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Champions League Tournament Simulation')
    parser.add_argument('--animate', action='store_true',
                       help='Generate animations for matches (will make simulation much slower)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations of tournament results')
    parser.add_argument('--group-duration', type=int, default=30,
                       help='Duration of group stage matches in minutes (defaults to 30 for faster simulation)')
    parser.add_argument('--knockout-duration', type=int, default=45,
                       help='Duration of knockout stage matches in minutes (defaults to 45 for faster simulation)')
    parser.add_argument('--use-enhanced-movement', action='store_true', default=True,
                       help='Use enhanced movement logic for player positioning (default)')
    parser.add_argument('--use-neural-movement', action='store_false', dest='use_enhanced_movement',
                       help='Use neural network model for player movement instead')
    
    args = parser.parse_args()
    
    # Set current directory to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tournament
    tournament_dir = run_tournament(args)
    
    if tournament_dir and args.visualize:
        print("\nVisualization Commands:")
        print(f"  - View Tournament Bracket:  open {tournament_dir}/tournament_bracket.png")
        print(f"  - View Group Standings:     open {tournament_dir}/group_standings.png")
        print(f"  - View Goal Statistics:     open {tournament_dir}/goal_statistics.png")

if __name__ == "__main__":
    main()
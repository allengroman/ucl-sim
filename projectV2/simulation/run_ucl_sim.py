#!/usr/bin/env python3
"""
Champions League Tournament Simulation Runner

This script provides a simple command-line interface for running
the Champions League tournament simulation with various options.
"""
import os
import sys
import subprocess
import argparse
import time
import datetime

def main():
    """Run a Champions League tournament simulation"""
    print("\n🏆 Champions League Tournament Simulator 🏆\n")
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run Champions League Tournament Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--teams', type=int, default=32,
                       help='Number of teams in the tournament')
    parser.add_argument('--animate', action='store_true',
                       help='Generate animations for matches (will make simulation much slower)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations of tournament results')
    parser.add_argument('--group-duration', type=int, default=30,
                       help='Duration of group stage matches in minutes')
    parser.add_argument('--knockout-duration', type=int, default=45,
                       help='Duration of knockout stage matches in minutes')
    parser.add_argument('--use-enhanced-movement', action='store_true', default=True,
                       help='Use enhanced movement logic for player positioning')
    parser.add_argument('--use-neural-movement', action='store_false', dest='use_enhanced_movement',
                       help='Use neural network model for player movement instead')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output from the simulation')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with extra logging')
    parser.add_argument('--output-dir', type=str, default='tournament_results',
                       help='Base directory to save results')
    
    args = parser.parse_args()
    
    # Set current directory to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create timestamp for the tournament
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"tournament_{timestamp}")
    
    # Build command for tournament simulation
    cmd = ["python", "champions_league.py"]
    
    # Add arguments
    cmd.extend(["--teams", str(args.teams)])
    cmd.extend(["--group-duration", str(args.group_duration)])
    cmd.extend(["--knockout-duration", str(args.knockout_duration)])
    cmd.extend(["--output-dir", output_dir])
        
    if args.animate:
        cmd.append("--animate")
    
    # Add movement model selection
    if args.use_enhanced_movement:
        cmd.append("--use-enhanced-movement")
    else:
        cmd.append("--use-neural-movement")
    
    # Run tournament simulation
    print("Starting UCL tournament simulation...")
    print(f"Output will be saved to: {output_dir}")
    print("This may take a while, especially if animations are enabled.")
    
    start_time = time.time()
    
    try:
        if args.verbose or args.debug:
            # Run with standard output
            process = subprocess.run(cmd, check=True)
        else:
            # Run and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Print important parts of the output
            output_lines = result.stdout.split('\n')
            
            # Print header
            print("\n==================================")
            print("🏆 CHAMPIONS LEAGUE SIMULATION 🏆")
            print("==================================\n")
            
            # Print group stage results
            groups_start = -1
            for i, line in enumerate(output_lines):
                if "Group Stage Draw:" in line:
                    groups_start = i
                    break
            
            if groups_start > 0:
                # Print groups
                print("\nGroups:")
                for i in range(groups_start, min(groups_start + 40, len(output_lines))):
                    if "SIMULATING GROUP STAGE" in output_lines[i]:
                        break
                    print(output_lines[i])
            
            # Find tournament winner
            for line in output_lines:
                if "🏆 CHAMPIONS:" in line:
                    print("\n" + line)
                    break
        
        # Generate visualizations if requested
        if args.visualize and os.path.exists(output_dir):
            print("\nGenerating visualizations...")
            viz_cmd = ["python", "visualize_tournament.py", 
                      "--tournament-dir", output_dir]
            subprocess.run(viz_cmd, check=True)
        
        # Print final message
        print(f"\nTournament completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Time taken: {(time.time() - start_time) / 60:.1f} minutes")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tournament: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output:\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
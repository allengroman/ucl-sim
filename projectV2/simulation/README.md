# Soccer Simulation with Integrated Models

This directory contains a simulation system that combines:
- LEM (Large Events Model) for on-ball event prediction
- Off-ball Movement Model for player positioning

## Files

### Single Match Simulation
- **integrated_simulation.py**: Main simulation script for running soccer matches
- **train_offball_minimal.py**: Script for training the off-ball movement model
- **run_match.py**: Easy launcher for running individual matches

### Tournament Simulation
- **champions_league.py**: Simulates a complete Champions League tournament
- **visualize_tournament.py**: Creates visualizations of tournament results
- **run_ucl_sim.py**: Improved launcher for Champions League tournament simulations

## How to Use

### 1. Train the Off-ball Movement Model

```bash
# Train with synthetic data
python train_offball_minimal.py --epochs 20 --samples 5000

# The model will be saved to ../models/offball_movement_model_final_TIMESTAMP.pt
```

### 2. Run a Single Match Simulation

Using the launcher script:
```bash
# Interactive mode - select teams
python run_match.py

# Specify teams
python run_match.py --home Barcelona --away "Real Madrid" --animate

# List available teams
python run_match.py --list-teams
```

Using direct simulation:
```bash
# Basic simulation between two teams
python integrated_simulation.py --home Bournemouth --away Valencia

# Simulation with animation
python integrated_simulation.py --home Bournemouth --away Valencia --animate

# Specify which off-ball model to use
python integrated_simulation.py --home Bournemouth --away Valencia --offball_model "../models/offball_movement_model_final_TIMESTAMP.pt"

# Change match duration and interval
python integrated_simulation.py --home Bournemouth --away Valencia --duration 45 --interval 10
```

### 3. Run a Champions League Tournament

```bash
# Run a tournament with default settings (fast simulation)
python run_ucl_sim.py

# Run with match animations and visualizations
python run_ucl_sim.py --animate --visualize

# Customize match durations
python run_ucl_sim.py --group-duration 45 --knockout-duration 90 --visualize

# Show detailed simulation output
python run_ucl_sim.py --verbose

# Debug mode with extra logging
python run_ucl_sim.py --debug
```

The tournament follows the Champions League format:
- Group Stage: 8 groups of 4 teams each (top 2 from each group advance)
- Round of 16: Group winners play against runners-up from other groups
- Quarter-finals: 8 teams
- Semi-finals: 4 teams
- Final: 2 teams

## Command Line Arguments

### Single Match Simulation
- `--home`: Home team name (default: "Bournemouth")
- `--away`: Away team name (default: "Valencia")
- `--lem_model`: Path to trained LEM model (default: "../models/simple_lem_model.pt")
- `--offball_model`: Path to trained off-ball movement model (default: "../models/offball_movement_model_final.pt")
- `--players`: Path to players CSV file (default: "../players.csv")
- `--teams`: Path to teams CSV file (default: "../teams.csv") 
- `--team_styles`: Path to team styles CSV file (default: "../team_styles.csv")
- `--duration`: Match duration in minutes (default: 90)
- `--interval`: Simulation interval in seconds (default: 5)
- `--width`: Field width in meters (default: 120)
- `--height`: Field height in meters (default: 80)
- `--animate`: Create animation of the match (flag)
- `--output`: Output directory for results and animations (default: "results")

### Tournament Simulation
- `--teams`: Number of teams in the tournament (default: 32)
- `--animate`: Generate animations for matches (makes simulation much slower)
- `--visualize`: Generate visualizations of tournament results
- `--group-duration`: Duration of group stage matches in minutes (default: 30)
- `--knockout-duration`: Duration of knockout stage matches in minutes (default: 45)
- `--use-enhanced-movement`: Use enhanced movement logic (default)
- `--use-neural-movement`: Use neural network model for player movement
- `--verbose`: Show detailed output from the simulation
- `--debug`: Run in debug mode with extra logging
- `--output-dir`: Base directory to save results (default: "tournament_results")

## Outputs

### Single Match Simulation
- Event CSVs with all match events (passes, shots, goals, etc.)
- Position history CSVs with player positions at each timestep
- Match animation video (when --animate is used)

### Tournament Simulation
- Group stage results and standings
- Knockout stage match results
- Tournament bracket visualization
- Group standings visualization
- Goal statistics visualization
- Summary JSON file with all tournament data

All tournament outputs are saved to the `tournament_results/tournament_TIMESTAMP` directory.
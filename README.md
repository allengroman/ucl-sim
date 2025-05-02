# UEFA Champions League Simulation

A realistic simulation of football matches and tournaments using discrete-event simulation and machine learning to model player and team behavior.

## Overview

This project provides a comprehensive simulation system for football (soccer) matches and tournaments, with a focus on the UEFA Champions League. The simulation integrates neural network models for player movement and event prediction with realistic physics and tactics.

Key features include:
- Second-by-second simulation of football matches
- Neural network-based player movement prediction
- LEM (Large Events Model) for predicting on-ball events
- Full Champions League tournament simulation
- Visualization of match and tournament results
- Customizable team formations and tactical styles

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9+ (GPU recommended for faster training but not required)
- CUDA-capable GPU (optional, for faster neural network inference)
- 8GB RAM minimum (16GB recommended for tournament simulations)
- 2GB free disk space

### Dependencies

Install the required Python packages:

```bash
# Clone the repository
git clone https://github.com/allengroman/ucl-sim
cd ucl-sim
```

Main dependencies include:
- PyTorch (machine learning framework)
- NumPy (numerical computing)
- Pandas (data manipulation)
- Matplotlib (visualization)
- SQLite3 (database)

## Project Structure

The codebase is organized into several key directories:

- `projectV2/models/`: Machine learning models for player movement and event prediction
- `projectV2/simulation/`: Match and tournament simulation engines
- `projectV2/data/`: Team, player, and historical event data
- `projectV2/LargeEventsModel/`: Implementation of Large Events Model (LEM) for prediction
- `projectV2/tournament_results/`: Output from tournament simulations

## Components

### Machine Learning Models

The simulation uses two main models:

1. **LEM (Large Events Model)**: Predicts on-ball events including:
   - Event type (pass, shot, duel, etc.)
   - Spatial coordinates
   - Event outcome (success/failure)

2. **Off-Ball Movement Predictor**: Neural network that predicts player movement based on:
   - Current game state
   - Team tactical styles
   - Player attributes
   - Ball position

### Simulation Engine

The integrated simulation engine (`IntegratedMatchSimulator` class) controls the flow of the simulation:
- Advances match time second-by-second
- Updates all player positions using neural network predictions
- Uses LEM to predict ball events
- Handles game events and phases
- Manages match statistics and results

### Tournament Simulator

The Champions League Tournament simulator (`ChampionsLeagueTournament` class) provides:
- Group stage simulation with 8 groups of 4 teams
- Knockout phase simulation (Round of 16, Quarter-finals, Semi-finals, Final)
- Statistical tracking across the tournament
- Visualization of tournament results

## How to Use

### Setup and Data Preparation

Before running simulations, make sure data is properly set up:

```bash
# Navigate to the project directory
cd projectV2

# If using fresh data (optional)
python data/extract.py
```

### Pre-trained Models

The repository includes pre-trained models, but you can train your own:

```bash
# Train the off-ball movement model
cd projectV2
python simulation/train_offball_minimal.py --epochs 20 --samples 5000

# Train a simple LEM model
python simple_lem_train.py
```

### Running a Single Match

```bash
# Navigate to the simulation directory
cd projectV2/simulation

# Interactive mode - select teams
python run_match.py

# Specify teams with animation, though incomplete animation without ball and other important features
python run_match.py --home Barcelona --away "Real Madrid" --animate

# List available teams
python run_match.py --list-teams
```

### Running a Champions League Tournament

```bash
# Navigate to the simulation directory
cd projectV2/simulation

# Run a tournament with default settings (fast simulation)
python run_ucl_sim.py

# Run with match animations and visualizations
python run_ucl_sim.py --animate --visualize

# Customize match durations
python run_ucl_sim.py --group-duration 45 --knockout-duration 90 --visualize

# Show detailed simulation output
python run_ucl_sim.py --verbose
```

## Command Line Arguments

### Single Match Simulation
- `--home`: Home team name
- `--away`: Away team name
- `--animate`: Create animation of the match
- `--duration`: Match duration in minutes
- `--interval`: Simulation interval in seconds
- `--lem_model`: Path to trained LEM model (default: "../models/simple_lem_model.pt")
- `--offball_model`: Path to trained off-ball movement model (default: "../models/offball_movement_model_final.pt")

### Tournament Simulation
- `--teams`: Number of teams in the tournament
- `--animate`: Generate animations for matches
- `--visualize`: Generate visualizations of tournament results
- `--group-duration`: Duration of group stage matches in minutes
- `--knockout-duration`: Duration of knockout stage matches in minutes
- `--use-enhanced-movement`: Use enhanced movement logic (default)
- `--use-neural-movement`: Use neural network model for player movement
- `--debug`: Run in debug mode with extra logging

## Performance Considerations

- For faster simulations, disable animation (`--animate` flag)
- Reduce match duration for quicker results (`--duration` or `--group-duration`/`--knockout-duration`)
- On machines without a GPU, avoid neural network movement prediction when possible
- For tournament simulations, allocate at least 4GB of RAM

## Outputs

### Single Match Simulation
- Event CSVs with all match events (passes, shots, goals, etc.)
- Position history CSVs with player positions at each timestep
- Match animation video (when --animate is used)

### Tournament Simulation
- Group stage results and standings
- Knockout stage match results
- Tournament bracket visualization
- Summary JSON file with all tournament data

## Troubleshooting

- **ModuleNotFoundError**: Ensure you're running from the correct directory and have installed all dependencies
- **CUDA errors**: If using GPU, make sure your CUDA drivers are compatible with PyTorch version
- **Memory issues**: For large tournaments, reduce the number of teams or run on a machine with more RAM
- **Visualization errors**: Ensure matplotlib is properly installed

## References

This implementation is based on the following resources:
- UEFA Champions League official rules and format
- Research in sports analytics and football simulation
- Machine learning approaches to player movement and event prediction

## Acknowledgments

- This project was developed as part of a course on simulation and modeling

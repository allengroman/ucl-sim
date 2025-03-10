# UEFA Champions League Simulation

A realistic simulation of football matches and tournaments using discrete-event simulation and machine learning to model player and team behavior.

## Overview

This project aims to build a realistic simulation of the UEFA Champions League (UCL) that captures the dynamic nature of football matches. The simulation operates at a second-by-second level, modeling player actions, ball movement, and match events in detail.

Key features include:
- Second-by-second simulation of football matches
- Player decision-making using probability models (with ML integration planned)
- Realistic ball physics and player movement
- Comprehensive state tracking for players, teams, and matches
- Modeling of key football actions like passing, shooting, tackling, etc.
- Factors including player fatigue, positioning, and attributes

## Project Structure

The codebase is organized into several key modules:

- `states.py`: Defines all the state classes and enums used in the simulation
- `gamesim.py`: Contains the basic match simulation engine
- `predict.py`: Probability-based system for determining action outcomes
- `descmodel.py`: Skeleton for the machine learning component (to be implemented)
- `webscrapper.py`: Scrapes fbref.com for football data

## Components

### State Representation

The simulation uses a comprehensive state system to represent all entities:

- **Player State**: Position, velocity, stamina, attributes, current action, etc.
- **Ball State**: Position, velocity, height, possession status, current action
- **Team State**: Formation, tactics, roster, goals, possession status
- **Match State**: Clock, period, game phase, events, score

### Simulation Engine

The core engine (`SimpleMatchSimulator` class) controls the flow of the simulation:
- Advances match time second-by-second
- Updates all player positions and states
- Handles ball physics and possession
- Manages game events and phases
- Coordinates player decision-making and actions

### Action Outcome System

The `ActionOutcomePredictor` class uses probability-based models to determine the success or failure of player actions:
- Takes into account player attributes, match context, and physical factors
- Models passing, shooting, tackling, dribbling, and other key actions
- Provides detailed outcomes with different types of success/failure

### Player Decision AI (Planned)

The simulation will use machine learning to model player decision-making:
- Will predict the most likely action a player should take given the current state
- Will consider spatial relationships, player attributes, and tactical context
- Structure is in place with the skeleton framework

## Future Development

The project is under active development with several planned enhancements:

1. **Machine Learning Integration**:
   - Implement the ML player decision system
   - Train models on real-world football data
   - Refine player behavior for greater realism

2. **Tournament Simulation**:
   - Add group stage and knockout phase logic
   - Implement draws and tournament progression
   - Model home/away leg dynamics

3. **Enhanced Physics and Tactics**:
   - More sophisticated ball movement
   - Advanced tactical systems (pressing, possession styles)
   - Set piece specialization

4. **Visualization and Analysis**:
   - Data visualization of match statistics
   - Heat maps and event plotting
   - Performance analysis tools

## References

This implementation is based on the following resources:
- UEFA Champions League official rules and format
- Research in sports analytics and football simulation
- Discrete-event simulation methodology

## Acknowledgments

- This project was developed as part of a course on simulation and modeling

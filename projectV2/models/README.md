# Soccer Simulation Models

This directory contains machine learning models for soccer simulation:

## Main Models

- `simple_lem_model.pt`: The trained Simple Large Events Model used for event prediction
- `offball_movement_predictor.py`: Neural network architecture for predicting off-ball player movement

## Purpose

These models serve two distinct purposes in the soccer simulation:

1. **Event Prediction Model**: Predicts the next on-ball event, including:
   - Event type (pass, shot, duel, etc.)
   - Spatial coordinates (x, y positions)
   - Event outcome (success/failure)

2. **Off-Ball Movement Predictor**: Predicts the movement of all 21 players not in possession of the ball based on:
   - Current game state
   - Team tactical styles (formation, pressing style, etc.)
   - Player attributes
   - Ball position

## Model Architecture

### Event Prediction Model

Uses a multi-task learning approach with three output heads:
- Event type classification head (using softmax/log-softmax)
- Spatial coordinates prediction head (using sigmoid to normalize to [0,1])
- Outcome prediction head (binary classification with sigmoid)

### Off-Ball Movement Predictor

Uses a transformer-based architecture to model interactions between players:
- Encodes player attributes, team tactics, and positional information
- Uses self-attention to capture spatial relationships and tactical patterns
- Predicts the next position for each player based on the game context

## Training

To train the off-ball movement predictor, use:

```bash
python train_offball_model.py --epochs 20 --batch_size 32 --visualize
```

To use a pre-trained model:

```bash
python train_offball_model.py --pretrained models/offball_movement_model_TIMESTAMP.pt --visualize
```

## Input Data

The models use the following data sources:

1. `players.csv`: Player attributes like pace, strength, passing, etc.
2. `teams.csv`: Basic team information
3. `team_styles.csv`: Team tactical approaches including formation, pressing style, etc.
4. `events.csv`: Match event data for training sequences
#!/usr/bin/env python3
"""
Champions League Tournament Simulation

This script simulates a complete Champions League tournament:
- Group Stage (8 groups of 4 teams)
- Round of 16
- Quarter-finals
- Semi-finals
- Final

The simulation uses the integrated match engine to simulate each match.
"""
import os
import sys
import random
import pandas as pd
import numpy as np
import argparse
import datetime
import subprocess
import json
import glob
from collections import defaultdict

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # For access to models

# Import necessary functions
from soccer_data_functions import get_top_teams

class ChampionsLeagueTournament:
    """Simulates a complete Champions League tournament"""
    
    def __init__(self, num_teams=32, match_engine="integrated_simulation.py", 
                 data_dir="./data", output_dir="tournament_results", animate=False,
                 use_enhanced_movement=True):
        """
        Initialize the tournament.
        
        Args:
            num_teams: Number of teams in the tournament (must be 32)
            match_engine: Script to use for match simulation
            data_dir: Directory containing team and player data
            output_dir: Directory to save results
            animate: Whether to generate animations for matches
            use_enhanced_movement: Whether to use enhanced movement logic
        """
        self.num_teams = num_teams
        self.match_engine = match_engine
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.animate = animate
        self.use_enhanced_movement = use_enhanced_movement
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Tournament data
        self.teams = []
        self.groups = []
        self.knockout_matches = {
            'round_of_16': [],
            'quarter_finals': [],
            'semi_finals': [],
            'final': []
        }
        self.champions = None
        
        # Load team data
        self.load_teams()
        
    def load_teams(self):
        """Load top teams based on team_level"""
        try:
            # Use get_top_teams to get top teams sorted by level
            teams_df = get_top_teams(n=self.num_teams, 
                                     teams_file=f"{self.data_dir}/teams.csv", 
                                     team_styles_file=f"{self.data_dir}/team_styles.csv")
            
            # Extract team names and levels
            self.teams = []
            for _, team in teams_df.iterrows():
                self.teams.append({
                    'name': team['name'],
                    'level': team['team_level'],
                    'id': team['id']
                })
            
            print(f"Loaded {len(self.teams)} teams for the tournament")
            
            # If we don't have enough teams, duplicate some
            if len(self.teams) < self.num_teams:
                print(f"Warning: Only found {len(self.teams)} teams, need {self.num_teams}")
                print("Duplicating teams to reach required number")
                
                # Duplicate teams with slight variations until we have enough
                while len(self.teams) < self.num_teams:
                    # Pick a random team to duplicate
                    team = random.choice(self.teams)
                    # Create a slightly modified version
                    new_team = {
                        'name': f"{team['name']} B",
                        'level': max(1, min(99, team['level'] + random.randint(-5, 5))),
                        'id': team['id']
                    }
                    self.teams.append(new_team)
            
            # Ensure we have exactly the number of teams we need
            self.teams = self.teams[:self.num_teams]
            
            # Shuffle teams before creating groups
            random.shuffle(self.teams)
            
        except Exception as e:
            print(f"Error loading teams: {e}")
            # Create generic teams as fallback
            self.teams = []
            for i in range(self.num_teams):
                self.teams.append({
                    'name': f"Team {i+1}",
                    'level': random.randint(50, 90),
                    'id': i+1
                })
    
    def create_groups(self):
        """Create groups for the group stage"""
        # For Champions League format, we need 8 groups of 4 teams
        num_groups = 8
        teams_per_group = 4
        
        # Create pool of teams based on "pots" (seeding)
        # Sort teams by level to create pots
        sorted_teams = sorted(self.teams, key=lambda x: x['level'], reverse=True)
        
        # Create 4 pots of 8 teams each
        pots = []
        for i in range(4):
            start_idx = i * 8
            end_idx = start_idx + 8
            pots.append(sorted_teams[start_idx:end_idx])
        
        # Create groups by taking one team from each pot
        self.groups = []
        for i in range(num_groups):
            group = {
                'name': chr(65 + i),  # Group A, B, C, etc.
                'teams': []
            }
            
            # Add one team from each pot
            for pot in pots:
                # Get a random team from the pot
                team_idx = random.randrange(len(pot))
                team = pot.pop(team_idx)
                
                # Add team to group
                group['teams'].append({
                    'name': team['name'],
                    'level': team['level'],
                    'id': team['id'],
                    'points': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                    'matches_played': 0,
                    'wins': 0,
                    'draws': 0,
                    'losses': 0
                })
            
            self.groups.append(group)
        
        # Print group assignments
        print("\nChampions League Group Stage Draw:")
        for group in self.groups:
            print(f"\nGroup {group['name']}:")
            for team in group['teams']:
                print(f"  - {team['name']} (Level: {team['level']})")
    
    def simulate_group_stage(self, match_duration=45):
        """Simulate the group stage matches"""
        print("\n===== SIMULATING GROUP STAGE =====")
        
        # Create directory for group stage results
        group_dir = os.path.join(self.output_dir, "group_stage")
        os.makedirs(group_dir, exist_ok=True)
        
        # Store all match results
        all_matches = []
        
        # For each group
        for group in self.groups:
            group_name = group['name']
            print(f"\nSimulating Group {group_name} matches:")
            
            # Create a mini round-robin tournament within the group
            for i, team1 in enumerate(group['teams']):
                for j, team2 in enumerate(group['teams']):
                    # Skip if same team
                    if i == j:
                        continue
                    
                    # Simulate match
                    result = self.simulate_match(
                        team1['name'], 
                        team2['name'], 
                        stage=f"Group {group_name}",
                        duration=match_duration
                    )
                    
                    # Update team statistics
                    self.update_team_stats(team1, team2, result)
                    
                    # Save match result
                    match_data = {
                        'group': group_name,
                        'home_team': team1['name'],
                        'away_team': team2['name'],
                        'home_goals': result['home_goals'],
                        'away_goals': result['away_goals'],
                        'home_shots': result.get('home_shots', 0),
                        'away_shots': result.get('away_shots', 0),
                        'winner': result['winner'],
                        'stage': f"Group {group_name}"
                    }
                    all_matches.append(match_data)
                    
                    # Print result
                    print(f"  {team1['name']} {result['home_goals']} - {result['away_goals']} {team2['name']}")
            
            # Sort teams in the group by points, then goal difference, then goals scored
            group['teams'] = sorted(
                group['teams'],
                key=lambda x: (x['points'], x['goals_for'] - x['goals_against'], x['goals_for']),
                reverse=True
            )
            
            # Print group standings
            print(f"\nGroup {group_name} Final Standings:")
            print(f"{'Team':<25} {'MP':>3} {'W':>3} {'D':>3} {'L':>3} {'GF':>3} {'GA':>3} {'GD':>3} {'Pts':>3}")
            for team in group['teams']:
                gd = team['goals_for'] - team['goals_against']
                print(f"{team['name']:<25} {team['matches_played']:>3} {team['wins']:>3} {team['draws']:>3} "
                      f"{team['losses']:>3} {team['goals_for']:>3} {team['goals_against']:>3} {gd:>3} {team['points']:>3}")
        
        # Save all group stage matches to CSV
        matches_df = pd.DataFrame(all_matches)
        matches_file = os.path.join(group_dir, "group_stage_matches.csv")
        matches_df.to_csv(matches_file, index=False)
        print(f"\nGroup stage results saved to {matches_file}")
        
        # Save group standings to CSV
        standings = []
        for group in self.groups:
            for position, team in enumerate(group['teams']):
                standing = {
                    'group': group['name'],
                    'position': position + 1,
                    'team': team['name'],
                    'points': team['points'],
                    'matches_played': team['matches_played'],
                    'wins': team['wins'],
                    'draws': team['draws'],
                    'losses': team['losses'],
                    'goals_for': team['goals_for'],
                    'goals_against': team['goals_against'],
                    'goal_difference': team['goals_for'] - team['goals_against']
                }
                standings.append(standing)
        
        standings_df = pd.DataFrame(standings)
        standings_file = os.path.join(group_dir, "group_standings.csv")
        standings_df.to_csv(standings_file, index=False)
        print(f"Group standings saved to {standings_file}")
        
        # Return qualified teams (top 2 from each group)
        qualified_teams = []
        for group in self.groups:
            qualified_teams.extend(group['teams'][:2])
        
        return qualified_teams
    
    def create_knockout_stage(self, qualified_teams):
        """Create the knockout stage fixtures"""
        print("\n===== KNOCKOUT STAGE DRAW =====")
        
        # For the Round of 16, Champions League uses a draw system
        # where group winners face runners-up from different groups
        
        # Separate group winners and runners-up
        group_winners = []
        group_runners_up = []
        
        for group in self.groups:
            group_winners.append(group['teams'][0])
            group_runners_up.append(group['teams'][1])
        
        # Shuffle runners-up
        random.shuffle(group_runners_up)
        
        # Create Round of 16 matches
        self.knockout_matches['round_of_16'] = []
        print("\nRound of 16 Draw:")
        for i, winner in enumerate(group_winners):
            runner_up = group_runners_up[i]
            match = {
                'home_team': winner,
                'away_team': runner_up,
                'home_goals': None,
                'away_goals': None,
                'winner': None
            }
            self.knockout_matches['round_of_16'].append(match)
            print(f"  {winner['name']} vs {runner_up['name']}")
    
    def simulate_knockout_stage(self, match_duration=90):
        """Simulate the knockout stage matches"""
        print("\n===== SIMULATING KNOCKOUT STAGE =====")
        
        # Create directory for knockout stage results
        knockout_dir = os.path.join(self.output_dir, "knockout_stage")
        os.makedirs(knockout_dir, exist_ok=True)
        
        # Stages to simulate
        stages = [
            ('round_of_16', 'Round of 16'),
            ('quarter_finals', 'Quarter-finals'),
            ('semi_finals', 'Semi-finals'),
            ('final', 'Final')
        ]
        
        for stage_key, stage_name in stages:
            print(f"\n{stage_name}:")
            
            # Skip if no matches in this stage
            if stage_key != 'round_of_16' and not self.knockout_matches[stage_key]:
                # Create matches for this stage based on winners from previous stage
                self.create_next_stage_matches(stage_key)
                if not self.knockout_matches[stage_key]:
                    break
            
            # Simulate matches in this stage
            winners = []
            all_matches = []
            
            for match in self.knockout_matches[stage_key]:
                # Get teams
                home_team = match['home_team']
                away_team = match['away_team']
                
                # Simulate match
                result = self.simulate_match(
                    home_team['name'], 
                    away_team['name'], 
                    stage=stage_name,
                    duration=match_duration
                )
                
                # Update match data
                match['home_goals'] = result['home_goals']
                match['away_goals'] = result['away_goals']
                match['winner'] = result['winner']
                
                # Handle ties in knockout stage (except final)
                if match['home_goals'] == match['away_goals'] and stage_key != 'final':
                    # Simulate penalty shootout
                    penalty_result = self.simulate_penalties(home_team['name'], away_team['name'])
                    match['penalties'] = penalty_result
                    match['winner'] = penalty_result['winner']
                    
                    # Print result with penalties
                    print(f"  {home_team['name']} {match['home_goals']} - {match['away_goals']} {away_team['name']} "
                          f"(Penalties: {penalty_result['home_score']} - {penalty_result['away_score']})")
                else:
                    # Print regular result
                    print(f"  {home_team['name']} {match['home_goals']} - {match['away_goals']} {away_team['name']}")
                
                # Add winner to next stage
                if match['winner'] == 'home':
                    winners.append(home_team)
                else:
                    winners.append(away_team)
                
                # Save match data
                match_data = {
                    'stage': stage_name,
                    'home_team': home_team['name'],
                    'away_team': away_team['name'],
                    'home_goals': match['home_goals'],
                    'away_goals': match['away_goals'],
                    'winner': match['winner'],
                    'penalties': 'Yes' if match.get('penalties') else 'No'
                }
                
                if match.get('penalties'):
                    match_data['home_penalties'] = match['penalties']['home_score']
                    match_data['away_penalties'] = match['penalties']['away_score']
                
                all_matches.append(match_data)
            
            # Save matches to CSV
            matches_df = pd.DataFrame(all_matches)
            matches_file = os.path.join(knockout_dir, f"{stage_key.lower()}.csv")
            matches_df.to_csv(matches_file, index=False)
            
            # Prepare for next stage
            if stage_key == 'final':
                # Tournament is over, set champions
                self.champions = winners[0]
                print(f"\n🏆 CHAMPIONS: {self.champions['name']} 🏆")
            else:
                # Set up next stage based on winners
                next_stage_key = stages[stages.index((stage_key, stage_name)) + 1][0]
                self.prepare_next_stage(next_stage_key, winners)
    
    def prepare_next_stage(self, stage_key, qualified_teams):
        """Prepare matches for the next stage"""
        # Reset matches for this stage
        self.knockout_matches[stage_key] = []
        
        # Shuffle teams for the draw
        random.shuffle(qualified_teams)
        
        # Create matches
        num_matches = len(qualified_teams) // 2
        
        print(f"\n{stage_key.replace('_', ' ').title()} Draw:")
        for i in range(num_matches):
            home_idx = i * 2
            away_idx = home_idx + 1
            
            home_team = qualified_teams[home_idx]
            away_team = qualified_teams[away_idx]
            
            match = {
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': None,
                'away_goals': None,
                'winner': None
            }
            
            self.knockout_matches[stage_key].append(match)
            print(f"  {home_team['name']} vs {away_team['name']}")
    
    def create_next_stage_matches(self, stage_key):
        """Create matches for the next stage based on previous stage results"""
        # Map stages to their previous stage
        stage_map = {
            'quarter_finals': 'round_of_16',
            'semi_finals': 'quarter_finals',
            'final': 'semi_finals'
        }
        
        prev_stage = stage_map.get(stage_key)
        if not prev_stage:
            return
        
        # Get winners from previous stage
        winners = []
        for match in self.knockout_matches[prev_stage]:
            if match['winner'] == 'home':
                winners.append(match['home_team'])
            else:
                winners.append(match['away_team'])
        
        # Prepare next stage
        self.prepare_next_stage(stage_key, winners)
    
    def simulate_match(self, home_team, away_team, stage="Group Stage", duration=90):
        """Simulate a match using the match engine"""
        print(f"  Simulating: {home_team} vs {away_team} ({stage})")
        
        try:
            # Build command - make sure to properly quote team names with spaces
            cmd = [
                "python", self.match_engine,
                "--home", f'"{home_team}"',  # Quote team names to handle spaces
                "--away", f'"{away_team}"',
                "--duration", str(duration),
                "--output", os.path.join(self.output_dir, "matches")
            ]
            
            # Add animation flag if requested
            if self.animate:
                cmd.append("--animate")
                
            # Add enhanced movement flag if enabled
            if self.use_enhanced_movement:
                cmd.append("--use-enhanced-movement")
            else:
                cmd.append("--use-neural-movement")
            
            # Run match simulation - using shell=True to ensure quotes are respected
            # First make sure the output directory exists
            matches_dir = os.path.join(self.output_dir, "matches")
            os.makedirs(matches_dir, exist_ok=True)
            
            # Debug output
            print(f"Running command: {' '.join(cmd)}")
            
            # Run the simulation
            process = subprocess.run(
                " ".join(cmd),  # Join as a string for shell execution
                shell=True,     # Use shell to handle quotes properly
                capture_output=True, 
                text=True, 
                check=False
            )
            
            # Parse output to get match result
            output = process.stdout
            
            # Debug output if no output was captured
            if not output:
                print(f"Warning: No output captured from subprocess.")
                print(f"Command exit code: {process.returncode}")
                print(f"stderr: {process.stderr}")
            else:
                # Print last few lines of output for debugging
                print("Last 5 lines of output:")
                output_lines = output.split('\n')
                for line in output_lines[-5:]:
                    print(f"  | {line}")
            
            # First try to find the result in the CSV files
            try:
                # Try to find the most recent match CSV file for these teams
                matches_dir = os.path.join(self.output_dir, "matches")
                if os.path.exists(matches_dir):
                    # Create clean team names for filename matching
                    home_team_filename = home_team.replace(' ', '_')
                    away_team_filename = away_team.replace(' ', '_')
                    
                    # Find matching CSV files - try both team orders
                    pattern1 = f"match_{home_team_filename}_vs_{away_team_filename}_*_events.csv"
                    pattern2 = f"match_{away_team_filename}_vs_{home_team_filename}_*_events.csv"
                    
                    matching_files1 = glob.glob(os.path.join(matches_dir, pattern1))
                    matching_files2 = glob.glob(os.path.join(matches_dir, pattern2))
                    
                    # Combine results and sort by modification time
                    matching_files = sorted(matching_files1 + matching_files2, key=os.path.getmtime, reverse=True)
                    
                    # Debug output
                    if matching_files:
                        print(f"Found {len(matching_files)} matching CSV files. Most recent: {os.path.basename(matching_files[0])}")
                    else:
                        print(f"No matching CSV files found for patterns: {pattern1} or {pattern2}")
                        
                        # Fallback: Try to find ANY events CSV file created in the last 2 minutes
                        # This helps when filenames don't match expected patterns (e.g., encoding issues, formatting differences)
                        all_event_files = glob.glob(os.path.join(matches_dir, "*_events.csv"))
                        recent_files = []
                        
                        for file in all_event_files:
                            # Check if file was created in last 2 minutes
                            file_time = os.path.getmtime(file)
                            if (datetime.datetime.now().timestamp() - file_time) < 120:  # 120 seconds = 2 minutes
                                recent_files.append(file)
                        
                        if recent_files:
                            # Sort by modification time (newest first)
                            matching_files = sorted(recent_files, key=os.path.getmtime, reverse=True)
                            print(f"Fallback: Found {len(matching_files)} recent event files. Using most recent: {os.path.basename(matching_files[0])}")
                    
                    if matching_files and os.path.exists(matching_files[0]):
                        # Found a matching events file, read it
                        events_df = pd.read_csv(matching_files[0])
                        
                        # Check if the team order is reversed in the CSV file
                        teams_reversed = False
                        if matching_files[0].find(f"{away_team_filename}_vs_{home_team_filename}") != -1:
                            teams_reversed = True
                            print(f"Teams appear in reverse order in CSV file. Adjusting extraction logic.")
                        
                        # Extract goals
                        home_goals = 0
                        away_goals = 0
                        
                        for _, event in events_df.iterrows():
                            if event['event_type'] == 'GOAL':
                                if (event['team'] == 'home' and not teams_reversed) or (event['team'] == 'away' and teams_reversed):
                                    home_goals += 1
                                else:
                                    away_goals += 1
                        
                        # Determine winner
                        if home_goals > away_goals:
                            winner = 'home'
                        elif away_goals > home_goals:
                            winner = 'away'
                        else:
                            winner = 'draw'
                        
                        # Count shots (accounting for possibly reversed teams)
                        if not teams_reversed:
                            home_shots = len(events_df[(events_df['team'] == 'home') & 
                                                      ((events_df['event_type'] == 'SHOT') | 
                                                       (events_df['event_type'] == 'GOAL'))])
                            away_shots = len(events_df[(events_df['team'] == 'away') & 
                                                      ((events_df['event_type'] == 'SHOT') | 
                                                       (events_df['event_type'] == 'GOAL'))])
                        else:
                            # If teams are reversed, swap home and away
                            home_shots = len(events_df[(events_df['team'] == 'away') & 
                                                      ((events_df['event_type'] == 'SHOT') | 
                                                       (events_df['event_type'] == 'GOAL'))])
                            away_shots = len(events_df[(events_df['team'] == 'home') & 
                                                      ((events_df['event_type'] == 'SHOT') | 
                                                       (events_df['event_type'] == 'GOAL'))])
                        
                        # Log extraction details for debugging
                        print(f"CSV extraction details:")
                        print(f"  File: {os.path.basename(matching_files[0])}")
                        print(f"  Teams reversed: {teams_reversed}")
                        print(f"  Goals: {home_team} {home_goals} - {away_goals} {away_team}")
                        print(f"  Shots: {home_team} {home_shots} - {away_shots} {away_team}")
                        
                        return {
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_goals': home_goals,
                            'away_goals': away_goals,
                            'home_shots': home_shots,
                            'away_shots': away_shots,
                            'winner': winner
                        }
            except Exception as e:
                print(f"Could not extract result from CSV: {e}")
            
            # Fall back to parsing the output
            result = self.parse_match_result(output, home_team, away_team)
            
            return result
            
        except Exception as e:
            print(f"Error simulating match: {e}")
            # Return a random result as fallback
            return self.generate_random_result(home_team, away_team)
    
    def parse_match_result(self, output, home_team, away_team):
        """Parse match engine output to extract the result"""
        try:
            if not output:
                print(f"No output from match simulation. Falling back to random result for {home_team} vs {away_team}")
                return self.generate_random_result(home_team, away_team)
                
            # Try to find the final score line
            score_line = None
            for line in reversed(output.split('\n')):
                if "Final Score:" in line:
                    score_line = line
                    break
            
            if score_line:
                # Extract goals more carefully
                try:
                    # The format is: Final Score: {home_team} {home_goals} - {away_goals} {away_team}
                    # We need to be smarter about parsing this
                    
                    # First split by "Final Score:" to get the score part
                    score_part = score_line.split('Final Score:')[1].strip()
                    
                    # Look for the pattern "X - Y" where X and Y are numbers
                    import re
                    score_match = re.search(r'(\d+)\s*-\s*(\d+)', score_part)
                    
                    if score_match:
                        home_goals = int(score_match.group(1))
                        away_goals = int(score_match.group(2))
                        
                        # Determine winner
                        if home_goals > away_goals:
                            winner = 'home'
                        elif away_goals > home_goals:
                            winner = 'away'
                        else:
                            winner = 'draw'
                        
                        # Try to extract shots
                        home_shots = 0
                        away_shots = 0
                        for line in reversed(output.split('\n')):
                            if "Shots:" in line:
                                try:
                                    shots_match = re.search(r'(\d+)\s*-\s*(\d+)', line.split('Shots:')[1].strip())
                                    if shots_match:
                                        home_shots = int(shots_match.group(1))
                                        away_shots = int(shots_match.group(2))
                                except:
                                    pass
                                break
                        
                        return {
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_goals': home_goals,
                            'away_goals': away_goals,
                            'home_shots': home_shots,
                            'away_shots': away_shots,
                            'winner': winner
                        }
                except Exception as e:
                    print(f"Error parsing score from '{score_line}': {e}")
                    # Continue to fallback below
            
            # Print debug info to help with troubleshooting
            print(f"Debug: Could not find 'Final Score:' in output. Using fallback.")
            # Look for another pattern: X - Y format anywhere in the output
            try:
                import re
                
                # First try exact team names
                for line in reversed(output.split('\n')):
                    # Look for lines containing team names and scores
                    if home_team in line and away_team in line:
                        score_match = re.search(r'(\d+)\s*-\s*(\d+)', line)
                        if score_match:
                            home_goals = int(score_match.group(1))
                            away_goals = int(score_match.group(2))
                            
                            # Determine winner
                            if home_goals > away_goals:
                                winner = 'home'
                            elif away_goals > home_goals:
                                winner = 'away'
                            else:
                                winner = 'draw'
                            
                            print(f"Found result using alternative method: {home_team} {home_goals} - {away_goals} {away_team}")
                            return {
                                'home_team': home_team,
                                'away_team': away_team,
                                'home_goals': home_goals,
                                'away_goals': away_goals,
                                'home_shots': 0,  # No shots data in this fallback
                                'away_shots': 0,
                                'winner': winner
                            }
                            
                # Fallback to just looking for any score pattern in the last few lines
                # This can help when the team names in the output don't exactly match
                print("Trying last resort score extraction...")
                last_20_lines = output.split('\n')[-20:]  # Check last 20 lines
                
                for line in reversed(last_20_lines):
                    # Look for the most common score reporting formats
                    if "Score:" in line or "score:" in line or "SCORE:" in line or " - " in line:
                        score_match = re.search(r'(\d+)\s*-\s*(\d+)', line)
                        if score_match:
                            home_goals = int(score_match.group(1))
                            away_goals = int(score_match.group(2))
                            
                            # Determine winner
                            if home_goals > away_goals:
                                winner = 'home'
                            elif away_goals > home_goals:
                                winner = 'away'
                            else:
                                winner = 'draw'
                            
                            print(f"Found result using last resort method: {home_goals} - {away_goals}")
                            print(f"Source line: '{line}'")
                            return {
                                'home_team': home_team,
                                'away_team': away_team,
                                'home_goals': home_goals,
                                'away_goals': away_goals,
                                'home_shots': 0,  # No shots data in this fallback
                                'away_shots': 0,
                                'winner': winner
                            }
            except Exception as e:
                print(f"Alternative parsing method failed: {e}")
            
            # If we couldn't parse the result, return a random one
            print(f"Falling back to random result for {home_team} vs {away_team}")
            return self.generate_random_result(home_team, away_team)
            
        except Exception as e:
            print(f"Error parsing match result: {e}")
            print(f"Falling back to random result for {home_team} vs {away_team}")
            return self.generate_random_result(home_team, away_team)
    
    def generate_random_result(self, home_team, away_team):
        """Generate a random match result as fallback, accounting for team strength"""
        # Find team data to get team levels
        home_team_data = next((team for team in self.teams if team['name'] == home_team), 
                              {'level': 75, 'name': home_team})
        away_team_data = next((team for team in self.teams if team['name'] == away_team), 
                              {'level': 75, 'name': away_team})
        
        # Use team levels to adjust scoring probabilities
        home_level = home_team_data['level'] / 100.0  # Normalize to 0-1
        away_level = away_team_data['level'] / 100.0  # Normalize to 0-1
        
        # Base rates adjusted by team quality with home advantage
        home_rate = 1.3 * (0.5 + 0.7 * home_level) * (1.0 + 0.5 * (home_level - away_level))
        away_rate = 1.0 * (0.5 + 0.7 * away_level) * (1.0 + 0.5 * (away_level - home_level))
        
        # Keep rates reasonable
        home_rate = max(0.4, min(2.3, home_rate))
        away_rate = max(0.3, min(1.8, away_rate))
        
        # Generate goals using Poisson distribution with team-adjusted rates
        home_goals = max(0, int(np.random.poisson(home_rate)))
        away_goals = max(0, int(np.random.poisson(away_rate)))
        
        # Log the result with team levels
        print(f"Random result: {home_team} (Lvl {home_team_data['level']}) {home_goals} - " +
              f"{away_goals} {away_team} (Lvl {away_team_data['level']})")
        print(f"  Scoring rates: Home {home_rate:.2f}, Away {away_rate:.2f}")
        
        # Determine winner
        if home_goals > away_goals:
            winner = 'home'
        elif away_goals > home_goals:
            winner = 'away'
        else:
            winner = 'draw'
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_shots': home_goals * 3 + random.randint(0, 5),
            'away_shots': away_goals * 3 + random.randint(0, 5),
            'winner': winner
        }
    
    def update_team_stats(self, home_team, away_team, result):
        """Update team statistics based on match result"""
        # Update matches played
        home_team['matches_played'] += 1
        away_team['matches_played'] += 1
        
        # Update goals
        home_team['goals_for'] += result['home_goals']
        home_team['goals_against'] += result['away_goals']
        away_team['goals_for'] += result['away_goals']
        away_team['goals_against'] += result['home_goals']
        
        # Update points and win/draw/loss records
        if result['winner'] == 'home':
            home_team['points'] += 3
            home_team['wins'] += 1
            away_team['losses'] += 1
        elif result['winner'] == 'away':
            away_team['points'] += 3
            away_team['wins'] += 1
            home_team['losses'] += 1
        else:  # Draw
            home_team['points'] += 1
            away_team['points'] += 1
            home_team['draws'] += 1
            away_team['draws'] += 1
    
    def simulate_penalties(self, home_team, away_team):
        """Simulate a penalty shootout"""
        # Each team takes 5 penalties initially
        home_score = 0
        away_score = 0
        
        for i in range(5):
            # Home team penalty
            if random.random() < 0.75:  # 75% chance to score
                home_score += 1
            
            # Away team penalty
            if random.random() < 0.75:  # 75% chance to score
                away_score += 1
            
            # Check if result is decided
            if (home_score > away_score + (5 - i)) or (away_score > home_score + (4 - i)):
                break
        
        # If still tied, sudden death
        if home_score == away_score:
            while True:
                # Home team penalty
                home_scores = random.random() < 0.75
                if home_scores:
                    home_score += 1
                
                # Away team penalty
                away_scores = random.random() < 0.75
                if away_scores:
                    away_score += 1
                
                # Check if we have a winner
                if home_scores and not away_scores:
                    break
                if away_scores and not home_scores:
                    break
        
        # Determine winner
        if home_score > away_score:
            winner = 'home'
        else:
            winner = 'away'
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'winner': winner
        }
    
    def save_tournament_summary(self):
        """Save a summary of the tournament"""
        # Create summary dictionary
        summary = {
            'champion': self.champions['name'],
            'groups': [],
            'knockout': {}
        }
        
        # Add group stage data
        for group in self.groups:
            group_data = {
                'name': group['name'],
                'teams': []
            }
            
            for team in group['teams']:
                team_data = {
                    'name': team['name'],
                    'points': team['points'],
                    'goal_difference': team['goals_for'] - team['goals_against']
                }
                group_data['teams'].append(team_data)
            
            summary['groups'].append(group_data)
        
        # Add knockout stage data
        for stage, matches in self.knockout_matches.items():
            summary['knockout'][stage] = []
            
            for match in matches:
                match_data = {
                    'home_team': match['home_team']['name'],
                    'away_team': match['away_team']['name'],
                    'result': f"{match['home_goals']} - {match['away_goals']}"
                }
                
                if match.get('penalties'):
                    match_data['penalties'] = f"{match['penalties']['home_score']} - {match['penalties']['away_score']}"
                
                summary['knockout'][stage].append(match_data)
        
        # Save to JSON
        summary_file = os.path.join(self.output_dir, "tournament_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTournament summary saved to {summary_file}")
    
    def run_tournament(self, group_match_duration=45, knockout_match_duration=90):
        """Run the complete tournament"""
        print("\n==================================")
        print("🏆 CHAMPIONS LEAGUE SIMULATION 🏆")
        print("==================================\n")
        
        # Create timestamp for this tournament
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_dir, f"tournament_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create groups
        self.create_groups()
        
        # Simulate group stage
        qualified_teams = self.simulate_group_stage(match_duration=group_match_duration)
        
        # Create knockout stage
        self.create_knockout_stage(qualified_teams)
        
        # Simulate knockout stage
        self.simulate_knockout_stage(match_duration=knockout_match_duration)
        
        # Save tournament summary
        self.save_tournament_summary()
        
        print("\nTournament simulation complete!")
        print(f"Results saved to {self.output_dir}")

def main():
    """Main function to run the tournament simulation"""
    parser = argparse.ArgumentParser(description='Run a Champions League tournament simulation')
    parser.add_argument('--teams', type=int, default=32,
                        help='Number of teams in the tournament (must be 32)')
    parser.add_argument('--match-engine', type=str, default='integrated_simulation.py',
                        help='Script to use for match simulation')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing team and player data')
    parser.add_argument('--output-dir', type=str, default='tournament_results',
                        help='Directory to save results')
    parser.add_argument('--animate', action='store_true',
                        help='Generate animations for matches')
    parser.add_argument('--group-duration', type=int, default=45,
                        help='Duration of group stage matches in minutes')
    parser.add_argument('--knockout-duration', type=int, default=90,
                        help='Duration of knockout stage matches in minutes')
    parser.add_argument('--use-enhanced-movement', action='store_true', default=True,
                        help='Use enhanced movement logic for player positioning')
    parser.add_argument('--use-neural-movement', action='store_false', dest='use_enhanced_movement',
                        help='Use neural network model for player movement instead')
    
    args = parser.parse_args()
    
    # Create and run tournament
    tournament = ChampionsLeagueTournament(
        num_teams=args.teams,
        match_engine=args.match_engine,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        animate=args.animate,
        use_enhanced_movement=args.use_enhanced_movement
    )
    
    tournament.run_tournament(
        group_match_duration=args.group_duration,
        knockout_match_duration=args.knockout_duration
    )

if __name__ == "__main__":
    main()
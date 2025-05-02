#!/usr/bin/env python3
"""
Visualize Champions League Tournament Results

This script creates visualizations of the tournament results.
"""
import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def load_tournament_results(tournament_dir):
    """Load tournament results from the output directory"""
    # Load tournament summary
    summary_file = os.path.join(tournament_dir, "tournament_summary.json")
    if not os.path.exists(summary_file):
        print(f"Tournament summary file not found: {summary_file}")
        return None
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load group stage results
    group_dir = os.path.join(tournament_dir, "group_stage")
    group_matches_file = os.path.join(group_dir, "group_stage_matches.csv")
    group_standings_file = os.path.join(group_dir, "group_standings.csv")
    
    if os.path.exists(group_matches_file):
        group_matches = pd.read_csv(group_matches_file)
    else:
        group_matches = None
    
    if os.path.exists(group_standings_file):
        group_standings = pd.read_csv(group_standings_file)
    else:
        group_standings = None
    
    # Load knockout stage results
    knockout_dir = os.path.join(tournament_dir, "knockout_stage")
    knockout_stages = [
        'round_of_16',
        'quarter_finals',
        'semi_finals',
        'final'
    ]
    
    knockout_results = {}
    for stage in knockout_stages:
        stage_file = os.path.join(knockout_dir, f"{stage}.csv")
        if os.path.exists(stage_file):
            knockout_results[stage] = pd.read_csv(stage_file)
        else:
            knockout_results[stage] = None
    
    return {
        'summary': summary,
        'group_matches': group_matches,
        'group_standings': group_standings,
        'knockout_results': knockout_results
    }

def create_tournament_bracket(results, output_file=None):
    """Create a visualization of the tournament bracket"""
    # Extract data
    summary = results['summary']
    knockout = summary['knockout']
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Setup for drawing the bracket
    stages = ['round_of_16', 'quarter_finals', 'semi_finals', 'final']
    num_teams = [16, 8, 4, 2]
    x_positions = [0, 3, 6, 9]
    
    # Find champion
    champion = summary.get('champion', 'Unknown')
    
    # Draw connecting lines for the bracket
    for stage_idx, stage in enumerate(stages[:-1]):
        matches = knockout.get(stage, [])
        next_stage = stages[stage_idx + 1]
        next_matches = knockout.get(next_stage, [])
        
        if not matches or not next_matches:
            continue
        
        # Draw connections between rounds
        curr_x = x_positions[stage_idx]
        next_x = x_positions[stage_idx + 1]
        
        teams_per_match = num_teams[stage_idx] // (num_teams[stage_idx] // 2)
        next_teams_per_match = num_teams[stage_idx+1] // (num_teams[stage_idx+1] // 2)
        
        spacing = 10 / (num_teams[stage_idx] / 2)
        next_spacing = 10 / (num_teams[stage_idx+1] / 2)
        
        for i, match in enumerate(matches):
            if i % teams_per_match == 0:
                # Get winner of this match
                winner = match.get('home_team') if match.get('result', '').split(' - ')[0] > match.get('result', '').split(' - ')[1] else match.get('away_team')
                
                # Get next match this winner goes to
                next_match_idx = i // teams_per_match
                if next_match_idx < len(next_matches):
                    # Draw a line from this match to the next one
                    y1 = 5 + i * spacing
                    y2 = 5 + next_match_idx * next_spacing
                    plt.plot([curr_x + 2, next_x], [y1, y2], 'k-', lw=1)
    
    # Draw teams and results
    for stage_idx, stage in enumerate(stages):
        matches = knockout.get(stage, [])
        if not matches:
            continue
        
        x = x_positions[stage_idx]
        spacing = 10 / (num_teams[stage_idx] / 2)
        
        for i, match in enumerate(matches):
            y = 5 + i * spacing
            
            # Draw match info
            home_team = match.get('home_team', 'Unknown')
            away_team = match.get('away_team', 'Unknown')
            result = match.get('result', '? - ?')
            
            # Determine winner for coloring
            if ' - ' in result:
                home_goals, away_goals = map(int, result.split(' - '))
                if home_goals > away_goals:
                    winner = home_team
                elif away_goals > home_goals:
                    winner = away_team
                else:
                    # If there's a penalty shootout
                    if 'penalties' in match:
                        pen_result = match['penalties']
                        home_pens, away_pens = map(int, pen_result.split(' - '))
                        winner = home_team if home_pens > away_pens else away_team
                    else:
                        winner = None
            else:
                winner = None
            
            # Format match text
            home_color = 'green' if winner == home_team else 'black'
            away_color = 'green' if winner == away_team else 'black'
            
            plt.text(x, y-0.4, home_team, fontsize=9, ha='left', va='center', color=home_color)
            plt.text(x, y+0.4, away_team, fontsize=9, ha='left', va='center', color=away_color)
            plt.text(x+1.5, y, result, fontsize=9, ha='center', va='center')
    
    # Draw champion
    plt.text(x_positions[-1] + 2, 5, f"🏆 {champion}", fontsize=14, ha='left', va='center', 
             color='gold', bbox=dict(facecolor='navy', alpha=0.2, boxstyle='round,pad=0.5'))
    
    # Add titles for each stage
    for stage_idx, stage in enumerate(stages):
        stage_name = stage.replace('_', ' ').title()
        plt.text(x_positions[stage_idx] + 1, 12, stage_name, fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='lightgray', alpha=0.3))
    
    # Set plot properties
    plt.title('Champions League Tournament Bracket', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Tournament bracket saved to {output_file}")
    
    return plt.gcf()

def create_group_standings_visualization(results, output_file=None):
    """Create a visualization of the group standings"""
    # Get group standings
    group_standings = results['group_standings']
    if group_standings is None:
        print("Group standings data not available")
        return None
    
    # Number of groups
    groups = sorted(group_standings['group'].unique())
    n_groups = len(groups)
    
    # Create figure
    fig, axs = plt.subplots(n_groups // 2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    for i, group in enumerate(groups):
        # Get data for this group
        group_data = group_standings[group_standings['group'] == group]
        
        # Sort by position
        group_data = group_data.sort_values('position')
        
        # Create table
        ax = axs[i]
        ax.axis('tight')
        ax.axis('off')
        
        # Table data
        table_data = []
        columns = ['Team', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
        
        for _, row in group_data.iterrows():
            team_data = [
                row['team'],
                row['matches_played'],
                row['wins'],
                row['draws'],
                row['losses'],
                row['goals_for'],
                row['goals_against'],
                row['goal_difference'],
                row['points']
            ]
            table_data.append(team_data)
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )
        
        # Set properties
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color qualifying teams
        for j in range(len(table_data)):
            if j < 2:  # Top 2 teams qualify
                for k in range(len(columns)):
                    table[(j+1, k)].set_facecolor('#d8f3dc')
        
        # Add group title
        ax.set_title(f'Group {group}', fontsize=12)
    
    # Set overall title
    plt.suptitle('Champions League Group Stage Standings', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Group standings visualization saved to {output_file}")
    
    return fig

def create_goal_statistics(results, output_file=None):
    """Create visualizations of goal statistics"""
    # Get match data
    group_matches = results['group_matches']
    knockout_results = results['knockout_results']
    
    if group_matches is None and all(v is None for v in knockout_results.values()):
        print("Match data not available")
        return None
    
    # Combine all match data
    all_matches = []
    
    if group_matches is not None:
        all_matches.append(group_matches)
    
    for stage, data in knockout_results.items():
        if data is not None:
            all_matches.append(data)
    
    if not all_matches:
        return None
    
    matches_df = pd.concat(all_matches, ignore_index=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Goals per stage
    ax1 = fig.add_subplot(gs[0, 0])
    
    stage_goals = matches_df.groupby('stage').apply(
        lambda x: x['home_goals'].sum() + x['away_goals'].sum()
    ).reset_index()
    stage_goals.columns = ['Stage', 'Goals']
    
    # Sort stages in tournament order
    stage_order = {
        'Group A': 0, 'Group B': 1, 'Group C': 2, 'Group D': 3,
        'Group E': 4, 'Group F': 5, 'Group G': 6, 'Group H': 7,
        'Round of 16': 8, 'Quarter-finals': 9, 'Semi-finals': 10, 'Final': 11
    }
    
    # Sort by custom order if available
    if all(stage in stage_order for stage in stage_goals['Stage']):
        stage_goals['Order'] = stage_goals['Stage'].map(stage_order)
        stage_goals = stage_goals.sort_values('Order')
    
    bars = ax1.bar(stage_goals['Stage'], stage_goals['Goals'], color='skyblue')
    ax1.set_title('Goals per Stage')
    ax1.set_ylabel('Number of Goals')
    ax1.tick_params(axis='x', rotation=90)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.0f}', ha='center', va='bottom')
    
    # 2. Goals distribution (home vs away)
    ax2 = fig.add_subplot(gs[0, 1])
    
    home_goals = matches_df['home_goals'].sum()
    away_goals = matches_df['away_goals'].sum()
    
    ax2.pie([home_goals, away_goals], labels=['Home', 'Away'], autopct='%1.1f%%',
            startangle=90, colors=['#ff9999','#66b3ff'])
    ax2.set_title('Goals Distribution (Home vs Away)')
    
    # 3. Goals per team (top 10)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Get all teams
    all_teams = pd.concat([
        matches_df[['home_team', 'home_goals']].rename(columns={'home_team': 'team', 'home_goals': 'goals'}),
        matches_df[['away_team', 'away_goals']].rename(columns={'away_team': 'team', 'away_goals': 'goals'})
    ])
    
    # Group by team and sum goals
    team_goals = all_teams.groupby('team')['goals'].sum().reset_index()
    team_goals = team_goals.sort_values('goals', ascending=False).head(10)
    
    bars = ax3.bar(team_goals['team'], team_goals['goals'], color='lightgreen')
    ax3.set_title('Top 10 Teams by Goals Scored')
    ax3.set_ylabel('Number of Goals')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Goal statistics visualization saved to {output_file}")
    
    return fig

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize Champions League tournament results')
    parser.add_argument('--tournament-dir', type=str, required=True,
                       help='Directory containing tournament results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save visualizations (defaults to tournament directory)')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or args.tournament_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tournament results
    results = load_tournament_results(args.tournament_dir)
    if results is None:
        print(f"Could not load tournament results from {args.tournament_dir}")
        return
    
    # Create visualizations
    bracket_file = os.path.join(output_dir, "tournament_bracket.png")
    create_tournament_bracket(results, bracket_file)
    
    standings_file = os.path.join(output_dir, "group_standings.png")
    create_group_standings_visualization(results, standings_file)
    
    stats_file = os.path.join(output_dir, "goal_statistics.png")
    create_goal_statistics(results, stats_file)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
"""
Enhanced Off-Ball Movement Logic for Soccer Simulation

This module contains improved off-ball movement logic that considers:
1. Player attributes and quality
2. Team tactical styles (attacking vs defensive)
3. Team formations and structures
4. Game state and match context
"""

import numpy as np
import random
import math

class EnhancedMovement:
    """
    Enhanced movement logic for soccer simulation that incorporates
    team tactics, player attributes, and tactical formations.
    """
    
    def __init__(self, field_width=120, field_height=80):
        """Initialize the movement logic system"""
        self.field_width = field_width
        self.field_height = field_height
        self.half_width = field_width / 2
        self.half_height = field_height / 2
        
        # Formation positioning templates (normalized to [0,1])
        self.formation_templates = self._create_formation_templates()
        
    def _create_formation_templates(self):
        """Create template positions for common formations"""
        templates = {
            # Format: [positions_when_attacking, positions_when_defending]
            "4-4-2": [
                # Attacking positions (normalized to [0,1] in field space)
                [
                    [0.05, 0.5],  # GK
                    [0.15, 0.2],  # RB
                    [0.15, 0.4],  # CB
                    [0.15, 0.6],  # CB
                    [0.15, 0.8],  # LB
                    [0.4, 0.15],  # RM
                    [0.35, 0.35], # CM
                    [0.35, 0.65], # CM
                    [0.4, 0.85],  # LM
                    [0.7, 0.35],  # ST
                    [0.7, 0.65]   # ST
                ],
                # Defending positions (more compact)
                [
                    [0.05, 0.5],  # GK
                    [0.2, 0.3],   # RB
                    [0.15, 0.4],  # CB
                    [0.15, 0.6],  # CB
                    [0.2, 0.7],   # LB
                    [0.3, 0.25],  # RM
                    [0.25, 0.4],  # CM
                    [0.25, 0.6],  # CM
                    [0.3, 0.75],  # LM
                    [0.4, 0.4],   # ST
                    [0.4, 0.6]    # ST
                ]
            ],
            "4-3-3": [
                # Attacking positions
                [
                    [0.05, 0.5],  # GK
                    [0.15, 0.2],  # RB
                    [0.15, 0.4],  # CB
                    [0.15, 0.6],  # CB
                    [0.15, 0.8],  # LB
                    [0.3, 0.3],   # CM
                    [0.35, 0.5],  # CM
                    [0.3, 0.7],   # CM
                    [0.6, 0.2],   # RW
                    [0.7, 0.5],   # ST
                    [0.6, 0.8]    # LW
                ],
                # Defending positions
                [
                    [0.05, 0.5],  # GK
                    [0.2, 0.3],   # RB
                    [0.15, 0.4],  # CB
                    [0.15, 0.6],  # CB
                    [0.2, 0.7],   # LB
                    [0.25, 0.35], # CM
                    [0.3, 0.5],   # CM
                    [0.25, 0.65], # CM
                    [0.4, 0.3],   # RW
                    [0.4, 0.5],   # ST
                    [0.4, 0.7]    # LW
                ]
            ],
            "3-5-2": [
                # Attacking positions
                [
                    [0.05, 0.5],  # GK
                    [0.15, 0.3],  # CB
                    [0.15, 0.5],  # CB
                    [0.15, 0.7],  # CB
                    [0.35, 0.15], # RWB
                    [0.3, 0.35],  # CM
                    [0.35, 0.5],  # CM
                    [0.3, 0.65],  # CM
                    [0.35, 0.85], # LWB
                    [0.65, 0.35], # ST
                    [0.65, 0.65]  # ST
                ],
                # Defending positions
                [
                    [0.05, 0.5],  # GK
                    [0.15, 0.3],  # CB
                    [0.15, 0.5],  # CB
                    [0.15, 0.7],  # CB
                    [0.25, 0.2],  # RWB
                    [0.25, 0.35], # CM
                    [0.3, 0.5],   # CM
                    [0.25, 0.65], # CM
                    [0.25, 0.8],  # LWB
                    [0.4, 0.4],   # ST
                    [0.4, 0.6]    # ST
                ]
            ],
            "5-3-2": [
                # Attacking positions
                [
                    [0.05, 0.5],  # GK
                    [0.15, 0.15], # RWB
                    [0.15, 0.3],  # CB
                    [0.15, 0.5],  # CB
                    [0.15, 0.7],  # CB
                    [0.15, 0.85], # LWB
                    [0.35, 0.3],  # CM
                    [0.4, 0.5],   # CM
                    [0.35, 0.7],  # CM
                    [0.7, 0.35],  # ST
                    [0.7, 0.65]   # ST
                ],
                # Defending positions
                [
                    [0.05, 0.5],  # GK
                    [0.15, 0.15], # RWB
                    [0.15, 0.3],  # CB
                    [0.15, 0.5],  # CB
                    [0.15, 0.7],  # CB
                    [0.15, 0.85], # LWB
                    [0.25, 0.3],  # CM
                    [0.3, 0.5],   # CM
                    [0.25, 0.7],  # CM
                    [0.4, 0.4],   # ST
                    [0.4, 0.6]    # ST
                ]
            ],
            "4-2-3-1": [
                # Attacking positions
                [
                    [0.05, 0.5],  # GK
                    [0.15, 0.2],  # RB
                    [0.15, 0.4],  # CB
                    [0.15, 0.6],  # CB
                    [0.15, 0.8],  # LB
                    [0.3, 0.35],  # DM
                    [0.3, 0.65],  # DM
                    [0.5, 0.2],   # RM
                    [0.55, 0.5],  # CAM
                    [0.5, 0.8],   # LM
                    [0.7, 0.5]    # ST
                ],
                # Defending positions
                [
                    [0.05, 0.5],  # GK
                    [0.2, 0.3],   # RB
                    [0.15, 0.4],  # CB
                    [0.15, 0.6],  # CB
                    [0.2, 0.7],   # LB
                    [0.25, 0.4],  # DM
                    [0.25, 0.6],  # DM
                    [0.35, 0.3],  # RM
                    [0.4, 0.5],   # CAM
                    [0.35, 0.7],  # LM
                    [0.45, 0.5]   # ST
                ]
            ]
        }
        
        return templates
    
    def update_player_positions(self, home_players, away_players, home_positions, away_positions, 
                               ball_position, ball_team, home_style, away_style):
        """
        Update player positions based on ball position, team tactics, and player attributes.
        
        Args:
            home_players: DataFrame of home team players with attributes
            away_players: DataFrame of away team players with attributes  
            home_positions: Current positions of home team players (numpy array)
            away_positions: Current positions of away team players (numpy array)
            ball_position: Current ball position (numpy array)
            ball_team: Which team has the ball ('home' or 'away')
            home_style: Dictionary with home team tactical style
            away_style: Dictionary with away team tactical style
            
        Returns:
            Updated positions for home and away teams (numpy arrays)
        """
        # Get team formations
        home_formation = home_style.get('formation', '4-3-3')
        away_formation = away_style.get('formation', '4-3-3')
        
        # Get team tactical styles (attacking or defensive)
        home_attacking = self._get_attacking_tendency(home_style)
        away_attacking = self._get_attacking_tendency(away_style)
        
        # Get team aggression (pressing intensity)
        home_aggression = self._get_aggression(home_style)
        away_aggression = self._get_aggression(away_style)
        
        # Get team quality levels (0-1 scale)
        home_quality = home_style.get('team_level', 50) / 100.0
        away_quality = away_style.get('team_level', 50) / 100.0
        
        # Calculate "territorial advantage" based on team quality difference
        # Higher quality teams will generally play in more advanced positions
        quality_diff = home_quality - away_quality
        territorial_advantage = max(-0.15, min(0.15, quality_diff * 0.3))
        
        # Update home team positions
        home_positions = self._update_team_positions(
            home_players, home_positions, ball_position, 
            has_possession=(ball_team == 'home'),
            formation=home_formation,
            attacking_tendency=home_attacking,
            aggression=home_aggression,
            quality=home_quality,
            territorial_shift=territorial_advantage,
            is_home=True
        )
        
        # Update away team positions
        away_positions = self._update_team_positions(
            away_players, away_positions, ball_position,
            has_possession=(ball_team == 'away'),
            formation=away_formation,
            attacking_tendency=away_attacking,
            aggression=away_aggression,
            quality=away_quality,
            territorial_shift=-territorial_advantage,  # Opposite for away team
            is_home=False
        )
        
        return home_positions, away_positions
    
    def _get_attacking_tendency(self, team_style):
        """Calculate attacking tendency (0-1) based on team style"""
        # Default to moderate attacking tendency
        attacking = 0.5
        
        # Adjust based on possession style
        if team_style.get('possession_style') == 'Possession':
            attacking += 0.1
        elif team_style.get('possession_style') == 'Counter':
            attacking += 0.05
        elif team_style.get('possession_style') == 'Defensive':
            attacking -= 0.1
            
        # Adjust based on attacking mindset
        if team_style.get('attacking_mindset') == 'Attacking':
            attacking += 0.15
        elif team_style.get('attacking_mindset') == 'Defensive':
            attacking -= 0.15
            
        # Adjust based on defensive style
        if team_style.get('defensive_style') == 'High Press':
            attacking += 0.1
        elif team_style.get('defensive_style') == 'Low-Block':
            attacking -= 0.1
            
        # Ensure value stays in [0, 1] range
        return max(0.1, min(0.9, attacking))
    
    def _get_aggression(self, team_style):
        """Calculate pressing aggression (0-1) based on team style"""
        # Default to moderate aggression
        aggression = 0.5
        
        # Adjust based on defensive style
        if team_style.get('defensive_style') == 'High Press':
            aggression += 0.2
        elif team_style.get('defensive_style') == 'Low-Block':
            aggression -= 0.2
            
        # Adjust based on defensive line
        if team_style.get('defensive_line') == 'High':
            aggression += 0.15
        elif team_style.get('defensive_line') == 'Low':
            aggression -= 0.15
            
        # Ensure value stays in [0, 1] range
        return max(0.1, min(0.9, aggression))
    
    def _update_team_positions(self, players, positions, ball_position, has_possession,
                              formation, attacking_tendency, aggression, quality,
                              territorial_shift=0, is_home=True):
        """
        Update positions for all players on a team
        
        Args:
            players: DataFrame with player data
            positions: Current player positions
            ball_position: Current ball position
            has_possession: Whether this team has possession
            formation: Team formation (e.g., "4-3-3")
            attacking_tendency: How attacking the team is (0-1)
            aggression: How aggressively the team presses (0-1)
            quality: Team quality level (0-1)
            territorial_shift: Adjustment for field position based on quality difference
            is_home: Whether this is the home team
            
        Returns:
            Updated positions array
        """
        num_players = len(positions)
        updated_positions = positions.copy()
        
        # Get normalized ball position (0-1 in field coordinates)
        ball_x_norm = ball_position[0] / self.field_width
        ball_y_norm = ball_position[1] / self.field_height
        
        # Determine if ball is in attacking or defending third
        if is_home:
            defending_third = ball_x_norm < 0.33
            attacking_third = ball_x_norm > 0.67
        else:
            defending_third = ball_x_norm > 0.67
            attacking_third = ball_x_norm < 0.33
            
        # Get team formation positions
        template_idx = 0 if has_possession else 1  # Use attacking or defending template
        formation_key = formation if formation in self.formation_templates else "4-3-3"
        formation_positions = self.formation_templates[formation_key][template_idx]
        
        # Calculate team center based on ball position
        # Teams shift toward the ball but maintain some structure
        if has_possession:
            # When in possession, the team's center follows the ball more loosely
            center_x = ball_position[0] * 0.3 + (self.field_width * 0.5) * 0.7
            # Apply territorial shift based on team quality
            center_x += territorial_shift * self.field_width
        else:
            # When defending, team center is more focused on the ball
            ball_weight = min(0.5, 0.3 + aggression * 0.2)  # More aggressive = more ball focus
            center_x = ball_position[0] * ball_weight + (self.field_width * 0.5) * (1 - ball_weight)
            # Apply territorial shift based on team quality
            center_x += territorial_shift * self.field_width
        
        # Keep center within reasonable bounds
        center_x = max(self.field_width * 0.2, min(self.field_width * 0.8, center_x))
        center_y = self.field_height / 2  # Center vertical position
        
        # Calculate team width (how spread out the team is horizontally)
        # Attacking teams spread wider when attacking
        if has_possession:
            width_factor = 0.7 + attacking_tendency * 0.3  # More attacking = wider
        else:
            width_factor = 0.5 - aggression * 0.2  # More aggressive defending = more compact
        
        # Calculate team depth (how spread out the team is vertically)
        if has_possession:
            depth_factor = 0.6 + attacking_tendency * 0.4  # More attacking = more stretched
        else:
            depth_factor = 0.5 - aggression * 0.3  # More aggressive defending = more compact
        
        # Scale width and depth based on field position and game state
        if defending_third:
            # More compact when defending in own third
            width_factor *= 0.7
            depth_factor *= 0.6
        elif attacking_third and has_possession:
            # More expansive when attacking in opponent's third
            width_factor *= 1.2
            depth_factor *= 1.1
        
        # Apply team direction based on home/away
        direction = 1 if is_home else -1
        
        # Update each player's position
        for i in range(min(num_players, len(formation_positions))):
            # Get player attributes and position
            player_row = players.iloc[i] if i < len(players) else None
            
            if player_row is not None:
                # Get player speed and workrate
                pace = player_row.get('pace', 70) / 100.0 if 'pace' in player_row else 0.7
                stamina = player_row.get('stamina', 70) / 100.0 if 'stamina' in player_row else 0.7
                # Calculate movement speed based on player attributes and quality
                movement_speed = (0.7 * pace + 0.3 * stamina) * (0.7 + 0.3 * quality)
            else:
                # Default values if player data not available
                movement_speed = 0.7 * quality
            
            # Get template position for this player's role
            template_x, template_y = formation_positions[i]
            
            # Flip x-coordinate for away team
            if not is_home:
                template_x = 1 - template_x
            
            # Calculate target position based on team center and formation
            formation_width = self.field_width * width_factor
            formation_depth = self.field_height * depth_factor
            
            # Transform normalized template position to field coordinates
            if is_home:
                target_x = center_x + (template_x - 0.5) * formation_width
            else:
                target_x = center_x - (template_x - 0.5) * formation_width
                
            target_y = center_y + (template_y - 0.5) * formation_depth
            
            # Goalkeeper stays closer to goal
            if i == 0:  # Goalkeeper
                if is_home:
                    target_x = min(target_x, self.field_width * 0.15)
                    # Only come out if ball is very close to goal
                    if ball_x_norm < 0.2:
                        target_x = min(target_x, ball_position[0] * 0.8 + 5)
                else:
                    target_x = max(target_x, self.field_width * 0.85)
                    # Only come out if ball is very close to goal
                    if ball_x_norm > 0.8:
                        target_x = max(target_x, ball_position[0] * 0.8 + self.field_width * 0.15)
            
            # Current position
            curr_x, curr_y = positions[i]
            
            # Special behaviors based on player role and game state
            if i > 0:  # Not goalkeeper
                # Determine player's role based on position in formation
                if i <= 5:  # Defenders and defensive midfielders
                    # Stay more positionally disciplined
                    role_discipline = 0.8
                    # Track attackers near the goal
                    if not has_possession and defending_third:
                        # Find nearest opponent to mark
                        if is_home:
                            # Away team attackers
                            attackers = [j for j in range(len(players), len(positions)) if j > len(players) - 4]
                            if attackers:
                                nearest_idx = min(attackers, key=lambda j: 
                                                 ((positions[j][0] - curr_x)**2 + 
                                                  (positions[j][1] - curr_y)**2)**0.5)
                                attacker_pos = positions[nearest_idx]
                                # Move closer to attacker if they're nearby
                                dist_to_attacker = ((attacker_pos[0] - curr_x)**2 + 
                                                   (attacker_pos[1] - curr_y)**2)**0.5
                                if dist_to_attacker < 20:
                                    # Balance between marking and maintaining position
                                    target_x = target_x * 0.6 + attacker_pos[0] * 0.4
                                    target_y = target_y * 0.5 + attacker_pos[1] * 0.5
                        else:
                            # Home team attackers
                            attackers = [j for j in range(min(5, len(positions))) if j > 0]
                            if attackers:
                                nearest_idx = min(attackers, key=lambda j: 
                                                 ((positions[j][0] - curr_x)**2 + 
                                                  (positions[j][1] - curr_y)**2)**0.5)
                                attacker_pos = positions[nearest_idx]
                                # Move closer to attacker if they're nearby
                                dist_to_attacker = ((attacker_pos[0] - curr_x)**2 + 
                                                   (attacker_pos[1] - curr_y)**2)**0.5
                                if dist_to_attacker < 20:
                                    # Balance between marking and maintaining position
                                    target_x = target_x * 0.6 + attacker_pos[0] * 0.4
                                    target_y = target_y * 0.5 + attacker_pos[1] * 0.5
                elif i > len(positions) - 4:  # Attackers
                    # Make attacking runs
                    if has_possession and attacking_third:
                        # Make runs into the box
                        role_discipline = 0.5  # More freedom to make runs
                        
                        # Chance of making a run based on player attributes and team style
                        run_chance = (pace * 0.4 + attacking_tendency * 0.6) * 0.8
                        
                        if random.random() < run_chance:
                            # Make a run toward the goal
                            if is_home:
                                run_x = target_x + random.uniform(10, 20)
                                # Stay in opponent's half
                                run_x = max(run_x, self.half_width + 5)
                            else:
                                run_x = target_x - random.uniform(10, 20)
                                # Stay in opponent's half
                                run_x = min(run_x, self.half_width - 5)
                                
                            # Vary the run direction
                            run_y = target_y + random.uniform(-10, 10)
                            run_y = max(10, min(self.field_height - 10, run_y))
                            
                            # Update target with run position
                            target_x = run_x
                            target_y = run_y
                    else:
                        # When defending, attackers don't track back as much
                        role_discipline = 0.6
                else:  # Midfielders
                    # Balanced between positional discipline and freedom
                    role_discipline = 0.7
                    
                    # Midfielders provide passing options
                    if has_possession and ball_position[0] > self.field_width * 0.3 and ball_position[0] < self.field_width * 0.7:
                        # Create passing triangles around ball carrier
                        dist_to_ball = ((ball_position[0] - curr_x)**2 + 
                                      (ball_position[1] - curr_y)**2)**0.5
                        if dist_to_ball < 30:
                            # Find space away from other teammates
                            spacing_x = 0
                            spacing_y = 0
                            for j in range(len(positions)):
                                if j != i:
                                    dx = curr_x - positions[j][0]
                                    dy = curr_y - positions[j][1]
                                    dist = max(1, (dx**2 + dy**2)**0.5)
                                    if dist < 15:
                                        # Move away from nearby teammates
                                        spacing_x += dx / dist * (15 - dist) * 0.2
                                        spacing_y += dy / dist * (15 - dist) * 0.2
                            
                            # Apply spacing adjustment
                            target_x += spacing_x
                            target_y += spacing_y
            else:
                # Goalkeeper is very positionally disciplined
                role_discipline = 0.95
            
            # Add small random movement
            random_x = random.uniform(-2, 2)
            random_y = random.uniform(-2, 2)
            
            # Calculate direction and distance to target
            dx = target_x - curr_x
            dy = target_y - curr_y
            distance = max(0.1, (dx**2 + dy**2)**0.5)
            
            # Normalize direction
            if distance > 0:
                dx /= distance
                dy /= distance
            
            # Determine movement amount based on distance and speed
            max_movement = 3.0 * movement_speed
            movement_amount = min(distance, max_movement)
            
            # Calculate new position
            # Players move more quickly to maintain formation when out of position
            if distance > 20:
                # Faster movement when far from position
                new_x = curr_x + dx * movement_amount * 1.5
                new_y = curr_y + dy * movement_amount * 1.5
            else:
                new_x = curr_x + dx * movement_amount
                new_y = curr_y + dy * movement_amount
            
            # Add small random component and apply role discipline
            new_x = curr_x * (1 - role_discipline) + new_x * role_discipline + random_x * (1 - role_discipline)
            new_y = curr_y * (1 - role_discipline) + new_y * role_discipline + random_y * (1 - role_discipline)
            
            # Ensure position stays within field bounds
            new_x = max(0, min(self.field_width, new_x))
            new_y = max(0, min(self.field_height, new_y))
            
            # Update position
            updated_positions[i] = [new_x, new_y]
        
        return updated_positions
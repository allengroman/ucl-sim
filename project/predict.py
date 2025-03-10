# Statistics models to predict outcome during the game
# currently implimented with simple probabilities, future to use player stats to dictate success rates
import random
import numpy as np
from typing import Optional, Dict, Any, Tuple
from states import (
    Player, Team, Match, Ball, 
    PlayerAction, TeamPhase, MatchPeriod, GamePhase,
    PhysicalState, BallAction
)

class ActionOutcomePredictor:
    """
    Simple probability-based predictor for football action outcomes.
    Uses basic rules and randomization to determine success or failure.
    """
    
    def __init__(self):
        """Initialize the predictor"""
        # Base success probabilities for different actions
        self.base_probabilities = {
            # With ball
            PlayerAction.PASS: 0.85,            # Short pass (high success rate)
            PlayerAction.THROUGH_PASS: 0.60,    # Through pass (medium success rate)
            PlayerAction.CROSS: 0.40,           # Cross (lower success rate)
            PlayerAction.SHOOT: 0.25,           # Basic shot
            PlayerAction.DRIBBLE: 0.70,         # Basic dribble
            PlayerAction.HOLD: 0.90,            # Hold the ball
            PlayerAction.TURN: 0.85,            # Turn with the ball
            PlayerAction.SKILL_MOVE: 0.50,      # Skill move
            PlayerAction.CLEAR: 0.80,           # Clearing the ball
            
            # Defensive actions
            PlayerAction.TACKLE: 0.50,          # Basic tackle
            PlayerAction.INTERCEPT: 0.40,       # Intercepting a pass
            PlayerAction.BLOCK_SHOT: 0.30,      # Blocking a shot
            PlayerAction.BLOCK_CROSS: 0.35,     # Blocking a cross
            PlayerAction.PRESS: 0.65,           # Pressing opponent
            
            # Goalkeeper actions
            PlayerAction.SAVE_SHOT: 0.70,       # Basic save
            PlayerAction.COLLECT_CROSS: 0.75,   # Collecting a cross
            PlayerAction.RUSH_OUT: 0.50,        # Rushing out
            
            # Default for other actions
            "DEFAULT": 0.75
        }
    
    def predict_success(self, action: PlayerAction, player: Player, 
                       context: Dict[str, Any]) -> bool:
        """
        Predict if an action will be successful.
        
        Args:
            action: The action being attempted
            player: The player performing the action
            context: Additional context (distance, pressure, etc.)
            
        Returns:
            True if the action is successful, False otherwise
        """
        # Get base probability
        base_prob = self.base_probabilities.get(action, self.base_probabilities["DEFAULT"])
        
        # Apply modifiers based on player attributes
        attribute_modifier = self._get_attribute_modifier(action, player)
        
        # Apply modifiers based on context
        context_modifier = self._get_context_modifier(action, context)
        
        # Apply stamina/fatigue modifier
        stamina_modifier = self._get_stamina_modifier(player)
        
        # Calculate final probability
        final_probability = base_prob * attribute_modifier * context_modifier * stamina_modifier
        
        # Cap probability between 0.05 and 0.95 to avoid guarantees
        final_probability = max(0.05, min(0.95, final_probability))
        
        # Roll the dice
        return random.random() < final_probability
    
    def _get_attribute_modifier(self, action: PlayerAction, player: Player) -> float:
        """
        Get a modifier based on player's relevant attributes.
        
        Args:
            action: The action being performed
            player: The player performing the action
            
        Returns:
            A multiplier affecting the success probability
        """
        # Map actions to relevant attributes
        if action in [PlayerAction.PASS, PlayerAction.THROUGH_PASS, PlayerAction.CROSS]:
            relevant_attribute = 'passing'
        elif action == PlayerAction.SHOOT:
            relevant_attribute = 'shooting'
        elif action in [PlayerAction.DRIBBLE, PlayerAction.SKILL_MOVE, PlayerAction.TURN]:
            relevant_attribute = 'dribbling'
        elif action in [PlayerAction.TACKLE, PlayerAction.INTERCEPT, PlayerAction.BLOCK_SHOT, 
                      PlayerAction.BLOCK_CROSS, PlayerAction.PRESS]:
            relevant_attribute = 'defending'
        elif action in [PlayerAction.SAVE_SHOT, PlayerAction.COLLECT_CROSS, PlayerAction.RUSH_OUT]:
            relevant_attribute = 'reactions'
        else:
            # Default to average of key attributes
            attrs = ['passing', 'shooting', 'dribbling', 'defending']
            attrs = [a for a in attrs if a in player.attributes]
            if attrs:
                return sum(player.attributes.get(a, 70) for a in attrs) / (len(attrs) * 100)
            return 0.7  # Default modifier if no attributes found
        
        # Get the attribute value or default to 70
        attr_value = player.attributes.get(relevant_attribute, 70)
        
        # Convert to a modifier (0.5 to 1.5 range for 0 to 100 attribute)
        return 0.5 + (attr_value / 100)
    
    def _get_context_modifier(self, action: PlayerAction, context: Dict[str, Any]) -> float:
        """
        Get a modifier based on the context of the action.
        
        Args:
            action: The action being performed
            context: Context information (distance, pressure, etc.)
            
        Returns:
            A multiplier affecting the success probability
        """
        modifier = 1.0
        
        # Apply distance modifier for appropriate actions
        if 'distance' in context and action in [PlayerAction.PASS, PlayerAction.THROUGH_PASS, 
                                             PlayerAction.CROSS, PlayerAction.SHOOT]:
            distance = context['distance']
            
            if action == PlayerAction.PASS:
                # Short passes (0-15) are easier, long passes (15+) get harder with distance
                if distance <= 15:
                    modifier *= 1.0
                else:
                    modifier *= max(0.3, 1.0 - ((distance - 15) / 50))
            
            elif action == PlayerAction.THROUGH_PASS:
                # Through passes get harder with distance
                modifier *= max(0.4, 1.0 - (distance / 60))
            
            elif action == PlayerAction.CROSS:
                # Medium crosses (15-30) are optimal, shorter or longer are harder
                if 15 <= distance <= 30:
                    modifier *= 1.0
                else:
                    modifier *= max(0.5, 1.0 - (abs(distance - 22.5) / 30))
            
            elif action == PlayerAction.SHOOT:
                # Shots get much harder with distance
                modifier *= max(0.1, 1.0 - (distance / 30))
        
        # Apply pressure modifier
        if 'pressure' in context:
            pressure = context['pressure']  # 0 to 1 scale
            
            # Most actions are harder under pressure
            if action != PlayerAction.CLEAR:  # Clearing is less affected by pressure
                modifier *= max(0.5, 1.0 - (pressure * 0.5))
        
        # Apply angle modifier for shots
        if 'angle' in context and action == PlayerAction.SHOOT:
            angle = context['angle']  # 0 to 90 degrees (0 = straight on goal)
            
            # Shots get harder at acute angles
            angle_factor = max(0.1, 1.0 - (angle / 90))
            modifier *= angle_factor
        
        return modifier
    
    def _get_stamina_modifier(self, player: Player) -> float:
        """
        Get a modifier based on player's stamina/fatigue.
        
        Args:
            player: The player performing the action
            
        Returns:
            A multiplier affecting the success probability
        """
        # Use current stamina as a percentage
        stamina_percentage = player.current_stamina / 100.0
        
        # At full stamina, no penalty. At 0 stamina, performance is 70% of normal
        return 0.7 + (0.3 * stamina_percentage)
    
    def predict_pass_outcome(self, player: Player, target_player: Optional[Player], 
                           match: Match) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Predict the outcome of a pass.
        
        Args:
            player: The player making the pass
            target_player: The intended recipient (or None for a general direction pass)
            match: The current match state
            
        Returns:
            Tuple of (success, outcome_type, details)
        """
        if target_player is None:
            # No valid target - pass automatically fails
            return False, "no_target", {}
        
        # Calculate distance between players
        distance = np.linalg.norm(target_player.position - player.position)
        
        # Check if it's a through pass or regular pass
        is_through_pass = False
        if distance > 20 or (target_player.position[0] > player.position[0] + 10 and 
                           player.team == match.home_team) or \
                          (target_player.position[0] < player.position[0] - 10 and 
                           player.team == match.away_team):
            is_through_pass = True
        
        # Determine action type
        action = PlayerAction.THROUGH_PASS if is_through_pass else PlayerAction.PASS
        
        # Find nearest opponent to estimate pressure
        opponents = match.away_team.lineup if player.team == match.home_team else match.home_team.lineup
        if opponents:
            nearest_opponent_dist = min(np.linalg.norm(opp.position - player.position) for opp in opponents)
            pressure = max(0.0, min(1.0, 1.0 - (nearest_opponent_dist / 10)))
        else:
            pressure = 0.0
        
        # Check for interception possibility
        interception_chance = 0.0
        interceptor = None
        
        # Create a straight line from passer to target
        if distance > 0:
            direction = (target_player.position - player.position) / distance
            
            # Check if any opponent is close to the pass line
            for opponent in opponents:
                # Vector from passer to opponent
                passer_to_opp = opponent.position - player.position
                
                # Project this vector onto the pass direction
                projection_distance = np.dot(passer_to_opp, direction)
                
                # Only consider opponents in the pass path (between passer and target)
                if 0 < projection_distance < distance:
                    # Calculate how far the opponent is from the pass line
                    # First, find the point on the pass line closest to the opponent
                    closest_point = player.position + direction * projection_distance
                    
                    # Distance from opponent to the pass line
                    perpendicular_distance = np.linalg.norm(opponent.position - closest_point)
                    
                    # If opponent is close enough to intercept
                    if perpendicular_distance < 3.0:
                        # Calculate interception probability based on distance and defending skill
                        opp_intercept_prob = (1.0 - (perpendicular_distance / 3.0)) * \
                                            (opponent.attributes.get('defending', 70) / 100)
                        
                        # Take the highest interception chance
                        if opp_intercept_prob > interception_chance:
                            interception_chance = opp_intercept_prob
                            interceptor = opponent
        
        # Context for pass success prediction
        context = {
            'distance': distance,
            'pressure': pressure
        }
        
        # Check if pass is successful
        pass_success = self.predict_success(action, player, context)
        
        # Even if pass is successful, it might be intercepted
        if pass_success and random.random() < interception_chance:
            return False, "intercepted", {'interceptor': interceptor}
        
        if pass_success:
            return True, "completed", {'distance': distance}
        else:
            # Different failure types
            if random.random() < 0.7:
                return False, "misplaced", {'distance': distance}
            else:
                return False, "overhit", {'distance': distance}
    
    def predict_shot_outcome(self, player: Player, match: Match) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Predict the outcome of a shot.
        
        Args:
            player: The player taking the shot
            match: The current match state
            
        Returns:
            Tuple of (success, outcome_type, details)
        """
        # Determine which goal the player is shooting at
        if player.team == match.home_team:
            goal_position = np.array([100.0, 50.0])  # Away team's goal
        else:
            goal_position = np.array([0.0, 50.0])    # Home team's goal
        
        # Calculate distance to goal
        distance = np.linalg.norm(goal_position - player.position)
        
        # Calculate angle to goal
        # First, get vector from player to goal
        to_goal = goal_position - player.position
        
        # For angle calculation, we need to know which direction is "straight at goal"
        straight_vector = np.array([1.0, 0.0]) if player.team == match.home_team else np.array([-1.0, 0.0])
        
        # Calculate angle between vectors (in degrees)
        dot_product = np.dot(to_goal, straight_vector)
        magnitudes = np.linalg.norm(to_goal) * np.linalg.norm(straight_vector)
        
        if magnitudes > 0:
            cos_angle = dot_product / magnitudes
            # Clamp to valid range due to potential floating-point errors
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = np.arccos(cos_angle) * 180 / np.pi
        else:
            angle = 90  # Default to 90 degrees if vectors have zero magnitude
        
        # Find nearest opponent to estimate pressure
        opponents = match.away_team.lineup if player.team == match.home_team else match.home_team.lineup
        if opponents:
            nearest_opponent_dist = min(np.linalg.norm(opp.position - player.position) for opp in opponents)
            pressure = max(0.0, min(1.0, 1.0 - (nearest_opponent_dist / 5)))
        else:
            pressure = 0.0
        
        # Context for shot success prediction
        context = {
            'distance': distance,
            'angle': angle,
            'pressure': pressure
        }
        
        # Check if shot is on target
        on_target = self.predict_success(PlayerAction.SHOOT, player, context)
        
        if not on_target:
            # Shot is off target
            if random.random() < 0.6:
                return False, "wide", {'distance': distance, 'angle': angle}
            else:
                return False, "over", {'distance': distance, 'angle': angle}
        
        # Shot is on target, check if it's a goal or saved
        
        # Find goalkeeper
        if player.team == match.home_team:
            goalkeeper = next((p for p in match.away_team.lineup if p.assigned_position.name == "GK"), None)
        else:
            goalkeeper = next((p for p in match.home_team.lineup if p.assigned_position.name == "GK"), None)
        
        # If no goalkeeper, shot automatically scores
        if goalkeeper is None:
            return True, "goal", {'distance': distance, 'angle': angle}
        
        # Goalkeeper save probability
        # Base on distance, angle, and goalkeeper attributes
        save_context = {
            'distance': distance,
            'angle': angle
        }
        
        # The harder the shot is to take, the easier it is to save
        save_modifier = 1.0 + ((distance / 50) * 0.5) + ((angle / 90) * 0.5)
        
        # Create modified context for the save attempt
        save_context['modifier'] = save_modifier
        
        # Check if goalkeeper saves
        save_success = self.predict_success(PlayerAction.SAVE_SHOT, goalkeeper, save_context)
        
        if save_success:
            # Determine if it's a clean catch or a parry
            if random.random() < 0.7:
                return False, "saved", {'goalkeeper': goalkeeper, 'clean_catch': True}
            else:
                return False, "saved", {'goalkeeper': goalkeeper, 'clean_catch': False}
        else:
            # Goal!
            return True, "goal", {'distance': distance, 'angle': angle}
    
    def predict_tackle_outcome(self, player: Player, target_player: Player, 
                             match: Match) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Predict the outcome of a tackle.
        
        Args:
            player: The player making the tackle
            target_player: The player being tackled
            match: The current match state
            
        Returns:
            Tuple of (success, outcome_type, details)
        """
        # Calculate distance between players
        distance = np.linalg.norm(target_player.position - player.position)
        
        # If players are too far apart, tackle automatically fails
        if distance > 2.0:
            return False, "out_of_range", {'distance': distance}
        
        # Context for tackle success prediction
        context = {
            'distance': distance,
            'target_dribbling': target_player.attributes.get('dribbling', 70) / 100.0
        }
        
        # Check if tackle is successful
        tackle_success = self.predict_success(PlayerAction.TACKLE, player, context)
        
        if tackle_success:
            # Determine if it's a clean tackle or a foul
            foul_chance = 0.2  # 20% base chance of foul
            
            # Increase foul chance if tackling from behind
            # Calculate angle between players
            direction_vector = target_player.position - player.position
            target_orientation = np.array([1.0, 0.0])  # Assume player facing right
            
            dot_product = np.dot(direction_vector, target_orientation)
            magnitudes = np.linalg.norm(direction_vector) * np.linalg.norm(target_orientation)
            
            if magnitudes > 0:
                cos_angle = dot_product / magnitudes
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                # If tackling from behind (angle > 90), increase foul chance
                if angle > 90:
                    foul_chance += 0.3
            
            # Check for foul
            is_foul = random.random() < foul_chance
            
            if is_foul:
                # Determine if it's a card offense
                yellow_card_chance = 0.3  # 30% chance of yellow card for a foul
                
                # Check for yellow card
                is_yellow = random.random() < yellow_card_chance
                
                return False, "foul", {'is_yellow_card': is_yellow}
            else:
                # Clean tackle
                return True, "clean_tackle", {'possession_won': True}
        else:
            # Tackle failed - player dribbled past
            return False, "missed_tackle", {'beaten': True}
    
    def predict_dribble_outcome(self, player: Player, match: Match) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Predict the outcome of a dribble attempt.
        
        Args:
            player: The player attempting to dribble
            match: The current match state
            
        Returns:
            Tuple of (success, outcome_type, details)
        """
        # Find nearest opponent
        opponents = match.away_team.lineup if player.team == match.home_team else match.home_team.lineup
        
        nearest_opponent = None
        nearest_dist = float('inf')
        
        for opponent in opponents:
            dist = np.linalg.norm(opponent.position - player.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_opponent = opponent
        
        # Context for dribble success prediction
        context = {
            'pressure': max(0.0, min(1.0, 1.0 - (nearest_dist / 5))) if nearest_opponent else 0.0
        }
        
        # Check if dribble is successful
        dribble_success = self.predict_success(PlayerAction.DRIBBLE, player, context)
        
        if dribble_success:
            # Calculate new position after successful dribble
            # Move in attacking direction
            if player.team == match.home_team:
                new_position = player.position + np.array([3.0, random.uniform(-1.0, 1.0)])
            else:
                new_position = player.position + np.array([-3.0, random.uniform(-1.0, 1.0)])
                
            # Keep within pitch boundaries
            new_position[0] = max(0.0, min(100.0, new_position[0]))
            new_position[1] = max(0.0, min(100.0, new_position[1]))
            
            return True, "successful_dribble", {'new_position': new_position}
        else:
            # Dribble failed
            if nearest_opponent and nearest_dist < 2.0:
                # Lost possession to nearest opponent
                return False, "dispossessed", {'opponent': nearest_opponent}
            else:
                # Just lost control
                return False, "lost_control", {}


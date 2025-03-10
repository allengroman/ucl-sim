# handles the game simulation logic
import simpy
import random
import numpy as np
from typing import List, Optional
from states import (
    Player, Team, Match, Ball, 
    PlayerAction, TeamPhase, MatchPeriod, GamePhase,
    PhysicalState, BallAction
)

class SimpleMatchSimulator:
    """
    Simple simulation engine for a football match.
    """
    
    def __init__(self, match: Match):
        """
        Initialize the match simulator.
        
        Args:
            match: The Match object containing teams, players, and state
        """
        self.match = match
        self.env = simpy.Environment()
        
        # Configuration
        self.time_step = 1.0  # Simulate in 1-second increments
        
    def setup(self):
        """Set up the simulation process"""
        # Main match process
        self.env.process(self.match_process())
        
    def run(self, until=None):
        """
        Run the simulation until the specified time.
        
        Args:
            until: Time in seconds to run until, or None for full match
        """
        if until is None:
            # Run for full match (90 minutes)
            until = 90 * 60
            
        self.setup()
        self.env.run(until=until)
        
    def match_process(self):
        """
        Main process controlling the match progression.
        """
        # Initialize match (kickoff)
        self.match.game_phase = GamePhase.KICKOFF
        self.match.period = MatchPeriod.FIRST_HALF
        
        # Initial kickoff
        starting_team = random.choice([self.match.home_team, self.match.away_team])
        starting_player = random.choice(starting_team.lineup)
        self.match.switch_possession(starting_team, starting_player)
        
        print(f"Match started! {self.match.home_team.name} vs {self.match.away_team.name}")
        print(f"Kickoff by {starting_team.name} ({starting_player.name})")
        
        # Main loop - run until end of match
        while self.match.period != MatchPeriod.FULLTIME:
            # Process one second of match time
            yield self.env.timeout(self.time_step)
            
            # Update match clock
            self.match.clock += self.time_step
            
            # Check for period transitions
            self._check_period_transitions()
            
            # Update all players
            self._update_players()
            
            # Update ball
            self._update_ball()
            
            # Process events (goals, fouls, etc.)
            self._process_events()
            
            # Log current state (once per minute)
            if int(self.match.clock) % 60 == 0:
                minute = int(self.match.clock / 60)
                print(f"Minute {minute}: {self.match.home_team.goals_scored}-{self.match.away_team.goals_scored}")
        
        # Match ended
        print(f"Final score: {self.match.home_team.name} {self.match.home_team.goals_scored} - {self.match.away_team.goals_scored} {self.match.away_team.name}")
            
    def _check_period_transitions(self):
        """Check and handle transitions between match periods"""
        if self.match.period == MatchPeriod.FIRST_HALF and self.match.clock >= 45 * 60:
            # First half ended
            self.match.period = MatchPeriod.HALFTIME
            print("Halftime!")
            
        elif self.match.period == MatchPeriod.HALFTIME and self.match.clock >= 45 * 60 + 15 * 60:
            # Second half started
            self.match.period = MatchPeriod.SECOND_HALF
            self.match.clock = 45 * 60  # Reset to beginning of second half
            print("Second half started!")
            
            # Second half kickoff
            second_half_team = self.match.away_team if self.match.team_in_possession == self.match.home_team else self.match.home_team
            second_half_player = random.choice(second_half_team.lineup)
            self.match.switch_possession(second_half_team, second_half_player)
            
        elif self.match.period == MatchPeriod.SECOND_HALF and self.match.clock >= 90 * 60:
            # Match ended
            self.match.period = MatchPeriod.FULLTIME
            print("Full time!")
    
    def _update_players(self):
        """Update all player states"""
        all_players = self.match.home_team.lineup + self.match.away_team.lineup
        
        for player in all_players:
            # Skip if player is unavailable
            if not player.is_available():
                continue
                
            # Update position and physical state
            self._update_player_position(player)
            
            # Update stamina
            self._update_player_stamina(player)
            
            # Update distance to ball
            player.update_distance_to_ball(float(np.linalg.norm(player.position - self.match.ball.position)))
            
            # Make decision if player is available for action
            if player.available_for_action:
                self._player_decision(player)
    
    def _update_player_position(self, player: Player):
        """Update player position based on current velocity"""
        # Basic position update based on velocity
        player.update_position(self.time_step)
        
        # Boundary check - keep players on the field
        x, y = player.position
        player.position = np.array([
            max(0, min(100, x)),
            max(0, min(100, y))
        ])
    
    def _update_player_stamina(self, player: Player):
        """Update player stamina based on activity"""
        # Simplified stamina reduction
        if player.speed > 0:
            # Moving - reduce stamina based on speed
            stamina_reduction = 0.01 * (1.0 + player.speed / 5.0)
            player.reduce_stamina(float(stamina_reduction))
        else:
            # Standing still - recover stamina slightly
            player.recover_stamina(0.005)
    
    def _player_decision(self, player: Player):
        """
        Decide what action the player should take next.
        This is the placeholder for more complex AI/ML decision making.
        """
        # TODO: Implement player decision logic 
        # For now, just a placeholder that does nothing
        # AI and ML models need to be refined and finished before implementation here
        pass
    
    def _update_ball(self):
        """Update ball position and state"""
        # If possessed by a player, ball follows player
        if self.match.ball.possession_player is not None:
            self.match.ball.position = self.match.ball.possession_player.position.copy()
        else:
            # Ball moves according to its velocity
            self.match.ball.update_position(self.time_step)
            
            # Apply friction to slow ball down
            self.match.ball.velocity *= 0.95  # 5% slowdown per second
            
            # If ball is very slow, stop it
            if np.linalg.norm(self.match.ball.velocity) < 0.1:
                self.match.ball.velocity = np.array([0.0, 0.0])
        
        # Check for out of bounds
        x, y = self.match.ball.position
        if x < 0 or x > 100 or y < 0 or y > 100:
            # Ball went out of bounds
            self.match.ball.position = np.array([
                max(0, min(100, x)),
                max(0, min(100, y))
            ])
            self.match.ball.velocity = np.array([0.0, 0.0])
            
            # Handle out of bounds (simplified)
            # In a full implementation, would set appropriate game phase and handle throw-ins, etc.
    
    def _process_events(self):
        """Process match events like goals, fouls, etc."""
        # Simplified - just check for goals
        self._check_for_goals()
        
        # Other events like fouls, offsides, etc. would be implemented here
    
    def _check_for_goals(self):
        """Check if a goal has been scored"""
        # Basic goal detection - ball crosses goal line between posts
        x, y = self.match.ball.position
        
        # Simplified goal width (from y=45 to y=55)
        goal_min_y, goal_max_y = 45, 55
        
        # Check if ball crossed goal line
        if x <= 0 and goal_min_y <= y <= goal_max_y:
            # Goal for away team
            self.match.away_team.goals_scored += 1
            self.match.home_team.goals_conceded += 1
            print(f"GOAL! {self.match.away_team.name} scored! ({self.match.home_team.goals_scored}-{self.match.away_team.goals_scored})")
            self._reset_after_goal(self.match.home_team)
            
        elif x >= 100 and goal_min_y <= y <= goal_max_y:
            # Goal for home team
            self.match.home_team.goals_scored += 1
            self.match.away_team.goals_conceded += 1
            print(f"GOAL! {self.match.home_team.name} scored! ({self.match.home_team.goals_scored}-{self.match.away_team.goals_scored})")
            self._reset_after_goal(self.match.away_team)
    
    def _reset_after_goal(self, kickoff_team: Team):
        """Reset the match state after a goal"""
        # Place ball at center
        self.match.ball.position = np.array([50.0, 50.0])
        self.match.ball.velocity = np.array([0.0, 0.0])
        
        # Give kickoff to the team that conceded
        starting_player = random.choice(kickoff_team.lineup)
        self.match.switch_possession(kickoff_team, starting_player)
        
        # Set game phase to kickoff
        self.match.game_phase = GamePhase.KICKOFF

# this is not for future, just current temporary testing
def create_sample_match() -> Match:
    """
    Create a sample match with two teams for testing.
    
    Returns:
        A Match object with two teams and players
    """
    # Create teams
    home_team = Team("HOME", "FC Barcelona")
    away_team = Team("AWAY", "Bayern Munich")
    
    # Create match
    match = Match(home_team, away_team)
    
    # Create players for home team
    home_players = []
    for i in range(11):
        # Determine position based on index
        if i == 0:
            position_name = "GK"
        elif i <= 4:
            position_name = "DEF"
        elif i <= 8:
            position_name = "MID"
        else:
            position_name = "FWD"
            
        # Create player
        player = Player(f"H{i+1}", f"Home {position_name} {i+1}", getattr(PlayingPosition, position_name), home_team)
        
        # Set initial position based on role (simplified)
        if position_name == "GK":
            player.position = np.array([5.0, 50.0])
        elif position_name == "DEF":
            x = 20.0
            y = 20.0 + (i * 15)  # Spread defenders across the width
            player.position = np.array([x, y])
        elif position_name == "MID":
            x = 50.0
            y = 20.0 + ((i-4) * 15)  # Spread midfielders across the width
            player.position = np.array([x, y])
        else:  # FWD
            x = 80.0
            y = 35.0 + ((i-9) * 30)  # Spread forwards across the width
            player.position = np.array([x, y])
            
        # Random attributes (simplified)
        for attr in ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'stamina']:
            player.attributes[attr] = 70 + random.randint(-10, 10)
            
        home_players.append(player)
    
    # Create players for away team
    away_players = []
    for i in range(11):
        # Determine position based on index
        if i == 0:
            position_name = "GK"
        elif i <= 4:
            position_name = "DEF"
        elif i <= 8:
            position_name = "MID"
        else:
            position_name = "FWD"
            
        # Create player
        player = Player(f"A{i+1}", f"Away {position_name} {i+1}", getattr(PlayingPosition, position_name), away_team)
        
        # Set initial position based on role (simplified)
        if position_name == "GK":
            player.position = np.array([95.0, 50.0])
        elif position_name == "DEF":
            x = 80.0
            y = 20.0 + (i * 15)  # Spread defenders across the width
            player.position = np.array([x, y])
        elif position_name == "MID":
            x = 50.0
            y = 20.0 + ((i-4) * 15)  # Spread midfielders across the width
            player.position = np.array([x, y])
        else:  # FWD
            x = 20.0
            y = 35.0 + ((i-9) * 30)  # Spread forwards across the width
            player.position = np.array([x, y])
            
        # Random attributes (simplified)
        for attr in ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'stamina']:
            player.attributes[attr] = 70 + random.randint(-10, 10)
            
        away_players.append(player)
    
    # Set team lineups
    home_team.lineup = home_players
    away_team.lineup = away_players
    
    # Add all players to teams
    home_team.players = home_players.copy()
    away_team.players = away_players.copy()
    
    return match


def run_sample_simulation():
    """Run a sample match simulation"""
    match = create_sample_match()
    simulator = SimpleMatchSimulator(match)
    simulator.run()
    
    # Return results
    return {
        'home_score': match.home_team.goals_scored,
        'away_score': match.away_team.goals_scored,
        'home_team': match.home_team.name,
        'away_team': match.away_team.name
    }


if __name__ == "__main__":
    # Run a sample simulation
    results = run_sample_simulation()
    print("\nMatch Summary:")
    print(f"{results['home_team']} {results['home_score']} - {results['away_score']} {results['away_team']}")

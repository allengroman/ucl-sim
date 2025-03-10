# includes all the custom datatypes for different states for a match
from enum import Enum, auto
from typing import List, Optional
import numpy as np


class PhysicalState(Enum):
    """Physical state of the ball"""
    ON_GROUND = auto()
    IN_AIR_LOW = auto()
    IN_AIR_MEDIUM = auto()
    IN_AIR_HIGH = auto()
    STATIONARY = auto()
    MOVING = auto()


class BallAction(Enum):
    """Current action being performed with the ball"""
    PASSED = auto()
    SHOT = auto()
    CROSSED = auto()
    DRIBBLED = auto()
    CLEARED = auto()
    LOOSE = auto()
    HEADED = auto()
    SAVED = auto()
    BLOCKED = auto()
    STATIC = auto()  # For set pieces, etc.


class FieldZone(Enum):
    """Zones on the pitch for simplified positioning"""
    DEFENSIVE_THIRD = auto()
    MIDDLE_THIRD = auto()
    ATTACKING_THIRD = auto()
    
    LEFT_WING = auto()
    CENTER = auto()
    RIGHT_WING = auto()
    
    PENALTY_AREA = auto()
    GOAL_AREA = auto()
    CORNER_AREA = auto()


class PlayerAction(Enum):
    """Actions a player can perform"""
    # With ball
    DRIBBLE = auto()
    PASS = auto()
    THROUGH_PASS = auto()
    CROSS = auto()
    SHOOT = auto()
    HOLD = auto()
    TURN = auto()
    SKILL_MOVE = auto()
    CLEAR = auto()
    
    # Without ball, team in possession
    RUN_INTO_SPACE = auto()
    PROVIDE_SUPPORT = auto()
    OVERLAPPING_RUN = auto()
    UNDERLAPPING_RUN = auto()
    AWAIT_PASS = auto()
    CREATE_SPACE = auto()
    ATTACK_CROSS = auto()
    DROP_DEEP = auto()
    
    # Without possession (defending)
    TACKLE = auto()
    INTERCEPT = auto()
    BLOCK_SHOT = auto()
    BLOCK_CROSS = auto()
    MARK_PLAYER = auto()
    PRESS = auto()
    CLOSE_DOWN = auto()
    JOCKEY = auto()
    COVER_SPACE = auto()
    TRACK_RUN = auto()
    CONTAIN = auto()
    RECOVER_POSITION = auto()
    
    # Goalkeeper specific
    SAVE_SHOT = auto()
    COLLECT_CROSS = auto()
    DISTRIBUTE = auto()
    RUSH_OUT = auto()
    SET_DEFENSIVE_LINE = auto()
    SWEEP = auto()
    
    # Transition
    COUNTER_ATTACK = auto()
    COUNTER_PRESS = auto()
    TRANSITION_TO_DEFENSE = auto()
    TRANSITION_TO_ATTACK = auto()
    
    # Other
    IDLE = auto()  # Default state, not performing a specific action


class ActionPhase(Enum):
    """Phase of an action execution"""
    STARTING = auto()
    EXECUTING = auto()
    FINISHING = auto()


class PlayingPosition(Enum):
    """Player's assigned tactical position"""
    GK = auto()  # Goalkeeper
    CB = auto()  # Center back
    LB = auto()  # Left back
    RB = auto()  # Right back
    CDM = auto()  # Central defensive midfielder
    CM = auto()  # Central midfielder
    CAM = auto()  # Central attacking midfielder
    LM = auto()  # Left midfielder
    RM = auto()  # Right midfielder
    LW = auto()  # Left winger
    RW = auto()  # Right winger
    CF = auto()  # Center forward
    ST = auto()  # Striker


class TeamPhase(Enum):
    """Team's current phase of play"""
    ATTACKING = auto()
    DEFENDING = auto()
    TRANSITION_TO_ATTACK = auto()
    TRANSITION_TO_DEFENSE = auto()
    SET_PIECE_ATTACK = auto()
    SET_PIECE_DEFENSE = auto()


class InjuryStatus(Enum):
    """Player's injury status"""
    HEALTHY = auto()
    SLIGHT_INJURY = auto()
    SEVERE_INJURY = auto()


class MatchPeriod(Enum):
    """Current period of the match"""
    FIRST_HALF = auto()
    SECOND_HALF = auto()
    FIRST_EXTRA = auto()
    SECOND_EXTRA = auto()
    PENALTIES = auto()
    HALFTIME = auto()
    FULLTIME = auto()


class GamePhase(Enum):
    """Current phase of play in the match"""
    OPEN_PLAY = auto()
    SET_PIECE = auto()
    OUT_OF_PLAY = auto()
    GOAL_KICK = auto()
    THROW_IN = auto()
    CORNER = auto()
    FREE_KICK = auto()
    PENALTY = auto()
    KICKOFF = auto()


class SetPieceStatus(Enum):
    """Status of a set piece if applicable"""
    NONE = auto()
    PREPARING = auto()
    READY = auto()
    TAKEN = auto()


class Formation(Enum):
    """Common football formations"""
    F_4_4_2 = auto()
    F_4_3_3 = auto()
    F_4_2_3_1 = auto()
    F_3_5_2 = auto()
    F_5_3_2 = auto()
    F_4_5_1 = auto()
    F_3_4_3 = auto()


class Ball:
    """Represents the ball state in the simulation"""
    
    def __init__(self):
        # Core states
        self.position = np.array([50.0, 50.0])  # x, y coordinates (0-100 scale)
        self.height = 0.0  # Height above ground in meters
        self.velocity = np.array([0.0, 0.0])  # velocity vector (x, y)
        self.speed = 0.0  # magnitude of velocity
        
        # Status states
        self.possession_team = None  # Team object or None if loose ball
        self.possession_player = None  # Player object or None
        self.physical_state = PhysicalState.STATIONARY
        self.action = BallAction.STATIC
        self.zone = FieldZone.CENTER
        
    def update_position(self, dt: float):
        """Update ball position based on velocity and time step"""
        self.position += self.velocity * dt
        self.speed = np.linalg.norm(self.velocity)
        
        # Update zone based on position
        self._update_zone()
        
    def _update_zone(self):
        """Update the field zone based on current position"""
        x, y = self.position
        
        # Update third (vertical zones)
        if x < 33.3:
            self.zone = FieldZone.DEFENSIVE_THIRD
        elif x < 66.6:
            self.zone = FieldZone.MIDDLE_THIRD
        else:
            self.zone = FieldZone.ATTACKING_THIRD
            
        # We could add more detailed zone logic here
        
    def set_possession(self, team, player):
        """Set the team and player in possession of the ball"""
        self.possession_team = team
        self.possession_player = player
        
        if player is not None:
            # Update ball position to player position
            self.position = player.position.copy()
            self.velocity = np.array([0.0, 0.0])
            self.physical_state = PhysicalState.ON_GROUND
            self.action = BallAction.DRIBBLED
        
    def clear_possession(self):
        """Set the ball as not in possession of any team/player"""
        self.possession_team = None
        self.possession_player = None
        self.action = BallAction.LOOSE


class Player:
    """Represents a player in the simulation"""
    
    def __init__(self, player_id: str, name: str, position: PlayingPosition, team):
        self.player_id = player_id
        self.name = name
        self.assigned_position = position
        self.team = team
        
        # Physical attributes (could be loaded from player database)
        self.attributes = {
            'pace': 70,
            'shooting': 70,
            'passing': 70,
            'dribbling': 70,
            'defending': 70,
            'physical': 70,
            'stamina': 100,
            'agility': 70,
            'balance': 70,
            'reactions': 70,
            'ball_control': 70,
            'composure': 70
        }
        
        # Physical state
        self.position = np.array([50.0, 50.0])  # Default to center, would be set by formation
        self.orientation = 0.0  # Angle in radians
        self.velocity = np.array([0.0, 0.0])
        self.speed = 0.0
        self.acceleration = np.array([0.0, 0.0])
        
        # Possession state
        self.has_ball = False
        self.distance_to_ball = 0.0
        
        # Action state
        self.current_action = PlayerAction.IDLE
        self.action_phase = ActionPhase.STARTING
        self.action_target = None  # Could be player, position, or goal
        self.action_timer = 0.0
        self.available_for_action = True
        
        # Tactical state
        self.current_zone = FieldZone.CENTER
        self.marking_assignment = None  # Player being marked
        self.formation_position = None  # Position in current formation
        
        # Physical condition
        self.current_stamina = 100.0
        self.fatigue = 0.0
        self.injury_status = InjuryStatus.HEALTHY
        self.sprint_available = True
        
        # Card state
        self.yellow_cards = 0
        self.red_card = False

    def update_distance_to_ball(self, dis: float):
        self.distance_to_ball = dis
        
    def update_position(self, dt: float):
        """Update player position based on velocity and time step"""
        self.position += self.velocity * dt
        self.speed = np.linalg.norm(self.velocity)
        
        # Update zone based on position
        self._update_zone()
        
    def _update_zone(self):
        """Update the player's current zone based on position"""
        x, y = self.position
        
        # Update third (vertical zones)
        if x < 33.3:
            self.current_zone = FieldZone.DEFENSIVE_THIRD
        elif x < 66.6:
            self.current_zone = FieldZone.MIDDLE_THIRD
        else:
            self.current_zone = FieldZone.ATTACKING_THIRD
    
    def start_action(self, action: PlayerAction, target=None):
        """Start a new player action"""
        if not self.available_for_action:
            return False
            
        self.current_action = action
        self.action_phase = ActionPhase.STARTING
        self.action_target = target
        self.action_timer = 0.0
        self.available_for_action = False
        
        return True
        
    def update_action(self, dt: float):
        """Update the current action"""
        if self.current_action == PlayerAction.IDLE:
            self.available_for_action = True
            return
            
        self.action_timer += dt
        
        # Simple state machine for action phases
        if self.action_timer < 0.3:  # First 0.3 seconds
            self.action_phase = ActionPhase.STARTING
        elif self.action_timer < 0.7:  # Next 0.4 seconds
            self.action_phase = ActionPhase.EXECUTING
        else:
            self.action_phase = ActionPhase.FINISHING
            
            # If action is complete, reset to idle
            if self.action_timer >= 1.0:  # Action takes 1 second (simplification)
                self.current_action = PlayerAction.IDLE
                self.available_for_action = True
                
    def reduce_stamina(self, amount: float):
        """Reduce player's stamina"""
        self.current_stamina = max(0.0, self.current_stamina - amount)
        
        # Increase fatigue as stamina decreases
        self.fatigue = max(0.0, min(100.0, 100.0 - self.current_stamina))
        
        # Disable sprint if stamina too low
        if self.current_stamina < 20.0:
            self.sprint_available = False
            
    def recover_stamina(self, amount: float):
        """Recover player's stamina"""
        self.current_stamina = min(100.0, self.current_stamina + amount)
        
        # Decrease fatigue as stamina recovers
        self.fatigue = max(0.0, 100.0 - self.current_stamina)
        
        # Enable sprint if stamina high enough
        if self.current_stamina > 30.0:
            self.sprint_available = True
            
    def is_available(self):
        """Check if player is available to play"""
        return (self.injury_status != InjuryStatus.SEVERE_INJURY and 
                not self.red_card)


class Team:
    """Represents a team in the simulation"""
    
    def __init__(self, team_id: str, name: str):
        self.team_id = team_id
        self.name = name
        
        # Core states
        self.possession = False
        self.goals_scored = 0
        self.goals_conceded = 0
        
        # Team composition
        self.players = []  # List of all players
        self.lineup = []   # Starting 11
        self.bench = []    # Substitutes
        self.captain = None
        
        # Tactical states
        self.formation = Formation.F_4_3_3
        self.phase = TeamPhase.DEFENDING
        self.tactics = {
            'pressing_intensity': 70,  # 0-100
            'defensive_line_height': 50,  # 0-100
            'width': 50,  # 0-100 (narrow to wide)
            'tempo': 60,  # 0-100 (slow to fast)
            'passing_directness': 50,  # 0-100 (short to long)
            'attacking_style': 'balanced',  # possession, counter, direct, etc.
            'defensive_style': 'balanced'  # press, contain, drop, etc.
        }
        
        # Match management
        self.substitutions_made = 0
        self.substitutions_available = 3
        
    def add_player(self, player: Player):
        """Add a player to the team"""
        self.players.append(player)
        
    def select_lineup(self, starting_eleven: List[Player]):
        """Set the starting lineup"""
        self.lineup = starting_eleven
        self.bench = [p for p in self.players if p not in starting_eleven]
        
    def set_formation(self, formation: Formation):
        """Set the team's formation"""
        self.formation = formation
        
        # Position players according to formation
        self._position_players()
        
    def _position_players(self):
        """Position players on the field according to formation"""
        # This would contain logic to place players based on formation
        # For example, a 4-3-3 would have different positions than a 4-4-2
        pass
        
    def make_substitution(self, player_off: Player, player_on: Player) -> bool:
        """Substitute a player"""
        if (self.substitutions_made >= self.substitutions_available or
                player_off not in self.lineup or
                player_on not in self.bench):
            return False
            
        self.lineup.remove(player_off)
        self.bench.remove(player_on)
        self.lineup.append(player_on)
        self.bench.append(player_off)
        self.substitutions_made += 1
        
        return True
        
    def update_tactics(self, **kwargs):
        """Update team tactics"""
        for key, value in kwargs.items():
            if key in self.tactics:
                self.tactics[key] = value


class Match:
    """Represents a football match in the simulation"""
    
    def __init__(self, home_team: Team, away_team: Team):
        self.home_team = home_team
        self.away_team = away_team
        
        # Ball state
        self.ball = Ball()
        
        # Match state
        self.clock = 0.0  # Time in seconds
        self.period = MatchPeriod.FIRST_HALF
        self.game_phase = GamePhase.KICKOFF
        self.set_piece_status = SetPieceStatus.NONE
        
        # Team in possession
        self.team_in_possession = None
        
        # Event history
        self.events = []
        
    def get_current_minute(self) -> int:
        """Get the current minute of the match"""
        return int(self.clock / 60.0)
        
    def advance_time(self, dt: float):
        """Advance the match clock by dt seconds"""
        self.clock += dt
        
        # Check for period transitions
        if self.period == MatchPeriod.FIRST_HALF and self.clock >= 45 * 60:
            self.period = MatchPeriod.HALFTIME
        elif self.period == MatchPeriod.HALFTIME and self.clock >= 45 * 60 + 15 * 60:
            self.period = MatchPeriod.SECOND_HALF
            self.clock = 45 * 60  # Reset to beginning of second half
        elif self.period == MatchPeriod.SECOND_HALF and self.clock >= 90 * 60:
            self.period = MatchPeriod.FULLTIME
            
    def switch_possession(self, team: Team, player: Optional[Player] = None):
        """Switch possession to the specified team and player"""
        self.team_in_possession = team
        team.possession = True
        
        # The other team loses possession
        other_team = self.away_team if team == self.home_team else self.home_team
        other_team.possession = False
        
        # Update ball possession
        self.ball.set_possession(team, player)
        
        if player:
            player.has_ball = True
            
    def record_event(self, event_type: str, player=None, team=None, details=None):
        """Record a match event"""
        event = {
            'time': self.clock,
            'minute': self.get_current_minute(),
            'type': event_type,
            'player': player.name if player else None,
            'team': team.name if team else None,
            'details': details or {}
        }
        
        self.events.append(event)
        
        # Special handling for goals
        if event_type == 'goal':
            if team == self.home_team:
                self.home_team.goals_scored += 1
                self.away_team.goals_conceded += 1
            else:
                self.away_team.goals_scored += 1
                self.home_team.goals_conceded += 1



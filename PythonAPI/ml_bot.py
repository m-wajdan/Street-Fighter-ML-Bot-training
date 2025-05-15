import tensorflow as tf
import joblib
import numpy as np
from command import Command
from buttons import Buttons

class MLBot:
    def __init__(self, model_path='fighting_game_model.keras',
                 stats_path='preprocessing_stats.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocess_stats = joblib.load(stats_path)
        self.command = Command()
        self.last_action = None
        self.action_history = []
        self.last_opponent_health = None
        self.last_bot_health = None
        self.attack_count = 0
        self.successful_attacks = 0
        self.current_sequence = []
        self.sequence_index = 0
        self.action_counts = {i: 0 for i in range(10)}  # Track action frequency
        self.recent_actions = []  # Track last 5 actions

        # Actions: expanded with new attacks
        self.action_map = {
            0: ["<", "-", "v+<", "-", "v", "-", ">+Y", "-", "!>+!Y"],  # Fireball
            1: [">+^+B", "-", "!>+!^+!B"],  # Spinning attack
            2: ["<+^+Y", "-", "!<+!^+!Y"],  # Jump attack (Y)
            3: [">+Y", "-", "!>+!Y"],  # Punch
            4: [">+B", "-", "!>+!B"],  # Kick
            5: ["<+^+B", "-", "!<+!^+!B"],  # Jump attack (B)
            6: ["Y", "-", "!Y"],  # Quick jab
            7: ["v+^+Y", "-", "!v+!^+!Y"],  # Low jump attack
            8: [">", "!>"],  # Advance right
            9: ["<", "!<"],  # Advance left
        }

        self.feature_order = [
            'timer', 'has_round_started', 'is_round_over',
            'player1_health', 'player1_x_coord', 'player1_y_coord',
            'player2_health', 'player2_x_coord', 'player2_y_coord',
            'player1_jumping', 'player1_crouching', 'player1_in_move',
            'player1_up', 'player1_down', 'player1_right', 'player1_left',
            'player2_jumping', 'player2_crouching', 'player2_in_move',
            'player2_up', 'player2_down', 'player2_right', 'player2_left',
            'player1_Y', 'player1_B', 'player1_X', 'player1_A',
            'player2_Y', 'player2_B', 'player2_X', 'player2_A',
            'distance', 'health_diff'
        ]

    def extract_features(self, game_state, player_num):
        p1 = game_state.player1
        p2 = game_state.player2

        features = {
            'timer': game_state.timer,
            'has_round_started': int(game_state.has_round_started),
            'is_round_over': int(game_state.is_round_over),
            'player1_health': p1.health,
            'player1_x_coord': p1.x_coord,
            'player1_y_coord': p1.y_coord,
            'player2_health': p2.health,
            'player2_x_coord': p2.x_coord,
            'player2_y_coord': p2.y_coord,
            'player1_jumping': int(p1.is_jumping),
            'player1_crouching': int(p1.is_crouching),
            'player1_in_move': int(p1.is_player_in_move),
            'player1_up': int(p1.player_buttons.up),
            'player1_down': int(p1.player_buttons.down),
            'player1_right': int(p1.player_buttons.right),
            'player1_left': int(p1.player_buttons.left),
            'player2_jumping': int(p2.is_jumping),
            'player2_crouching': int(p2.is_crouching),
            'player2_in_move': int(p2.is_player_in_move),
            'player2_up': int(p2.player_buttons.up),
            'player2_down': int(p2.player_buttons.down),
            'player2_right': int(p2.player_buttons.right),
            'player2_left': int(p2.player_buttons.left),
            'player1_Y': int(p1.player_buttons.Y),
            'player1_B': int(p1.player_buttons.B),
            'player1_X': int(p1.player_buttons.X),
            'player1_A': int(p1.player_buttons.A),
            'player2_Y': int(p2.player_buttons.Y),
            'player2_B': int(p2.player_buttons.B),
            'player2_X': int(p2.stage.player_buttons.X),
            'player2_A': int(p2.player_buttons.A),
            'distance': abs(p1.x_coord - p2.x_coord),
            'health_diff': p1.health - p2.health
        }

        return np.array([features.get(k, 0) for k in self.feature_order])

    def preprocess_state(self, game_state, player_num):
        try:
            features = self.extract_features(game_state, player_num)
            features = (features - self.preprocess_stats['mean']) / self.preprocess_stats['std']
            return features.reshape(1, -1)
        except Exception as e:
            print(f"Error in preprocess_state: {e}")
            return np.zeros((1, len(self.feature_order)))

    def execute_command(self, command, buttons):
        buttons.up = False
        buttons.down = False
        buttons.left = False
        buttons.right = False
        buttons.Y = False
        buttons.B = False
        buttons.X = False
        buttons.A = False
        buttons.R = False

        if command == "-":
            pass  # No-op for delay
        elif command == "<":
            buttons.left = True
        elif command == "!<":
            buttons.left = False
        elif command == ">":
            buttons.right = True
        elif command == "!>":
            buttons.right = False
        elif command == "v":
            buttons.down = True
        elif command == "!v":
            buttons.down = False
        elif command == "^":
            buttons.up = True
        elif command == "!^":
            buttons.up = False
        elif command == "v+<":
            buttons.down = True
            buttons.left = True
        elif command == "!v+!<":
            buttons.down = False
            buttons.left = False
        elif command == "v+>":
            buttons.down = True
            buttons.right = True
        elif command == "!v+!>":
            buttons.down = False
            buttons.right = False
        elif command == ">+Y":
            buttons.right = True
            buttons.Y = True
        elif command == "!>+!Y":
            buttons.right = False
            buttons.Y = False
        elif command == "<+Y":
            buttons.left = True
            buttons.Y = True
        elif command == "!<+!Y":
            buttons.left = False
            buttons.Y = False
        elif command == ">+^+B":
            buttons.right = True
            buttons.up = True
            buttons.B = True
        elif command == "!>+!^+!B":
            buttons.right = False
            buttons.up = False
            buttons.B = False
        elif command == "<+^+Y":
            buttons.left = True
            buttons.up = True
            buttons.Y = True
        elif command == "!<+!^+!Y":
            buttons.left = False
            buttons.up = False
            buttons.Y = False
        elif command == ">+B":
            buttons.right = True
            buttons.B = True
        elif command == "!>+!B":
            buttons.right = False
            buttons.B = False
        elif command == "<+^+B":
            buttons.left = True
            buttons.up = True
            buttons.B = True
        elif command == "!<+!^+!B":
            buttons.left = False
            buttons.up = False
            buttons.B = False
        elif command == "Y":
            buttons.Y = True
        elif command == "!Y":
            buttons.Y = False
        elif command == "v+^+Y":
            buttons.down = True
            buttons.up = True
            buttons.Y = True
        elif command == "!v+!^+!Y":
            buttons.down = False
            buttons.up = False
            buttons.Y = False

        print(f"Executing command: {command}, Buttons: up:{buttons.up}, down:{buttons.down}, left:{buttons.left}, right:{buttons.right}, Y:{buttons.Y}, B:{buttons.B}, X:{buttons.X}, A:{buttons.A}, R:{buttons.R}")

    def fight(self, game_state, player_num):
        try:
            features = self.preprocess_state(game_state, player_num)
            model_action_probs = self.model.predict(features, verbose=0)[0]
            # Softmax with temperature for diversity
            temperature = 1.5
            scaled_logits = model_action_probs / temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            softmax_probs = exp_logits / exp_logits.sum()
            # Penalize recent actions
            attack_probs = softmax_probs[[0, 1, 2, 3, 4, 5, 6, 7]].copy()
            for recent_action in self.recent_actions:
                if recent_action in [0, 1, 2, 3, 4, 5, 6, 7]:
                    attack_probs[recent_action] *= 0.5
            attack_probs = attack_probs / attack_probs.sum()
            print(f"Model probabilities: {model_action_probs}, Softmax probs: {softmax_probs}, Attack probs: {attack_probs}")
        except Exception as e:
            print(f"Error in model prediction: {e}")
            attack_probs = np.array([1/8]*8)  # Fallback: equal attack probs

        # Game state
        p1 = game_state.player1 if player_num == "1" else game_state.player2
        p2 = game_state.player2 if player_num == "1" else game_state.player1
        try:
            distance = abs(p1.x_coord - p2.x_coord)
        except Exception as e:
            print(f"Error calculating distance: {e}")
            distance = 0  # Assume close

        opponent_health = p2.health
        bot_health = p1.health

        # Track health changes
        if self.last_opponent_health is not None:
            damage_dealt = self.last_opponent_health - opponent_health
            print(f"Damage dealt: {damage_dealt}")
            if damage_dealt > 0 and self.last_action in [0, 1, 2, 3, 4, 5, 6, 7]:
                self.successful_attacks += 1
        self.last_opponent_health = opponent_health

        if self.last_bot_health is not None:
            damage_taken = self.last_bot_health - bot_health
            print(f"Damage taken: {damage_taken}")
        self.last_bot_health = bot_health

        # Log state
        bot_state = f"Jumping:{p1.is_jumping}, Crouching:{p1.is_crouching}, InMove:{p1.is_player_in_move}"
        opponent_state = f"Attacking:{p2.is_player_in_move}, Jumping:{p2.is_jumping}, Crouching:{p2.is_crouching}"
        opponent_buttons = f"Y:{p2.player_buttons.Y}, B:{p2.player_buttons.B}, X:{p2.player_buttons.X}, A:{p2.player_buttons.A}"
        print(f"Bot state: {bot_state}, Position: ({p1.x_coord}, {p1.y_coord})")
        print(f"Opponent state: {opponent_state}, Buttons: {opponent_buttons}, Position: ({p2.x_coord}, {p2.y_coord})")
        print(f"Distance: {distance}")

        # Sequence execution
        buttons = Buttons()
        if self.current_sequence and self.sequence_index < len(self.current_sequence):
            # Continue current sequence
            self.execute_command(self.current_sequence[self.sequence_index], buttons)
            self.sequence_index += 1
            if self.sequence_index >= len(self.current_sequence):
                self.current_sequence = []
                self.sequence_index = 0
                if self.last_action in [0, 1, 2, 3, 4, 5, 6, 7]:
                    self.attack_count += 1
        else:
            # Choose action
            attack_range = 100
            diff = p2.x_coord - p1.x_coord if player_num == "1" else p1.x_coord - p2.x_coord
            if distance < attack_range:
                action = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], p=attack_probs)
            else:
                action = 8 if diff > 0 else 9  # Advance
            self.current_sequence = self.action_map[action]
            self.sequence_index = 0
            self.execute_command(self.current_sequence[self.sequence_index], buttons)
            self.sequence_index += 1

        # Update action tracking
        self.last_action = action
        self.action_history.append(action)
        self.action_counts[action] += 1
        self.recent_actions.append(action)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)
        attack_success_rate = (self.successful_attacks / self.attack_count * 100) if self.attack_count > 0 else 0
        print(f"Action chosen: {self.get_last_action()}, Attack Count: {self.attack_count}, Attack Success Rate: {attack_success_rate:.2f}%")
        print(f"Sequence: {self.current_sequence}, Index: {self.sequence_index}")
        print(f"Action frequency: {self.action_counts}")
        print(f"Recent actions: {[self.get_last_action_name(a) for a in self.recent_actions]}")

        # Set buttons
        if player_num == "1":
            self.command.player_buttons = buttons
        else:
            self.command.player2_buttons = buttons

        return self.command

    def get_last_action(self):
        return {
            0: "Fireball",
            1: "Spinning Attack",
            2: "Jump Attack (Y)",
            3: "Punch",
            4: "Kick",
            5: "Jump Attack (B)",
            6: "Quick Jab",
            7: "Low Jump Attack",
            8: "Advance Right",
            9: "Advance Left"
        }.get(self.last_action, "Unknown")

    def get_last_action_name(self, action):
        return {
            0: "Fireball",
            1: "Spinning Attack",
            2: "Jump Attack (Y)",
            3: "Punch",
            4: "Kick",
            5: "Jump Attack (B)",
            6: "Quick Jab",
            7: "Low Jump Attack",
            8: "Advance Right",
            9: "Advance Left"
        }.get(action, "Unknown")
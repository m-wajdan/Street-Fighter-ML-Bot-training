from pynput import keyboard
from command import Command
from buttons import Buttons

class KeyboardController:
    def __init__(self):
        self.command = Command()
        self.listener = None
        
        # Map keyboard keys to button actions
        self.key_map = {
            # Movement
            keyboard.Key.left: ("left", True),
            keyboard.Key.right: ("right", True),
            keyboard.Key.up: ("up", True),
            keyboard.Key.down: ("down", True),
            
            # Attacks (mapped to ASDFQW)
            keyboard.KeyCode.from_char('a'): ("Y", True),
            keyboard.KeyCode.from_char('d'): ("B", True),
            keyboard.KeyCode.from_char('z'): ("X", True),
            keyboard.KeyCode.from_char('x'): ("A", True),
            keyboard.KeyCode.from_char('q'): ("L", True),
            keyboard.KeyCode.from_char('w'): ("R", True),
            
            # Key releases (same keys but False)
            (keyboard.Key.left, False): ("left", False),
            (keyboard.Key.right, False): ("right", False),
            (keyboard.Key.up, False): ("up", False),
            (keyboard.Key.down, False): ("down", False),
            (keyboard.KeyCode.from_char('a'), False): ("Y", False),
            (keyboard.KeyCode.from_char('d'), False): ("B", False),
            (keyboard.KeyCode.from_char('z'), False): ("X", False),
            (keyboard.KeyCode.from_char('x'), False): ("A", False),
            (keyboard.KeyCode.from_char('q'), False): ("L", False),
            (keyboard.KeyCode.from_char('w'), False): ("R", False),
        }
        self.start_listener()

    def start_listener(self):
        """Start non-blocking keyboard listener"""
        self.listener = keyboard.Listener(
            on_press=lambda key: self.handle_key(key, True),
            on_release=lambda key: self.handle_key(key, False)
        )
        self.listener.start()

    def handle_key(self, key, is_press):
        """Handle both key presses and releases"""
        lookup_key = key if is_press else (key, False)
        
        if lookup_key in self.key_map:
            button_name, button_state = self.key_map[lookup_key]
            setattr(self.command.player_buttons, button_name, button_state)

    def get_command(self, game_state, player_id):
        """Return the current command state"""
        return self.command
import socket
import numpy as np
import json
from game_state import GameState
import sys
import time
from ml_bot import MLBot
from command import Command

def connect(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print("Connected to game!")
    return client_socket

def send(client_socket, command):
    if not isinstance(command, Command):
        print("Invalid command object passed to send function")
        return
    command_dict = command.object_to_dict()
    payload = json.dumps(command_dict).encode()
    client_socket.sendall(payload)

def receive(client_socket):
    payload = client_socket.recv(4096)
    input_dict = json.loads(payload.decode())
    return GameState(input_dict)

def main():
    player_num = sys.argv[1]
    try:
        client_socket = connect(9999 if player_num == '1' else 10000)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    try:
        bot = MLBot(
            model_path='fighting_game_model.keras',
            stats_path='preprocessing_stats.pkl'
        )
        print("ML Bot initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize ML Bot: {e}")
        sys.exit(1)

    current_game_state = None

    try:
        while True:
            try:
                current_game_state = receive(client_socket)
                if int(current_game_state.is_round_over):
                    print("Round over, exiting.")
                

                bot_command = bot.fight(current_game_state, player_num)
                send(client_socket, bot_command)
                time.sleep(0.016)

            except Exception as e:
                print(f"Unexpected error in game loop: {e}")
                send(client_socket, Command())  # Safe fallback command
                continue

    except KeyboardInterrupt:
        print("Bot interrupted by user.")
    finally:
        client_socket.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()
import socket
import json
from game_state import GameState
#from bot import fight
import sys
from bot import Bot
import csv
import os
from keyboard_controller import KeyboardController
from ml_bot import MLBot
import pickle
import logging

def connect(port):
    #For making a connection with the game
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print ("Connected to game!")
    return client_socket

def send(client_socket, command):
    #This function will send your updated command to Bizhawk so that game reacts according to your command.
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    #receive the game state and return game state
    pay_load = client_socket.recv(4096)
    input_dict = json.loads(pay_load.decode())
    game_state = GameState(input_dict)

    return game_state

def writeGameState(current_game_state):
    row = []

    # Prepare data row
    row.extend([current_game_state.timer, current_game_state.fight_result, current_game_state.has_round_started, current_game_state.is_round_over])
    row.extend([current_game_state.player1.player_id, current_game_state.player1.health, current_game_state.player1.x_coord, current_game_state.player1.y_coord,
                current_game_state.player1.is_jumping, current_game_state.player1.is_crouching, current_game_state.player1.is_player_in_move, current_game_state.player1.move_id,
                current_game_state.player1.player_buttons.up, current_game_state.player1.player_buttons.down, current_game_state.player1.player_buttons.right,
                current_game_state.player1.player_buttons.left])
    row.extend([current_game_state.player2.player_id, current_game_state.player2.health, current_game_state.player2.x_coord, current_game_state.player2.y_coord,
                current_game_state.player2.is_jumping, current_game_state.player2.is_crouching, current_game_state.player2.is_player_in_move, current_game_state.player2.move_id,
                current_game_state.player2.player_buttons.up, current_game_state.player2.player_buttons.down, current_game_state.player2.player_buttons.right,
                current_game_state.player2.player_buttons.left,
                current_game_state.player1.player_buttons.Y, current_game_state.player1.player_buttons.B, current_game_state.player1.player_buttons.X,
                current_game_state.player1.player_buttons.A, current_game_state.player1.player_buttons.L, current_game_state.player1.player_buttons.R,
                current_game_state.player1.player_buttons.select, current_game_state.player1.player_buttons.start,
                current_game_state.player2.player_buttons.Y, current_game_state.player2.player_buttons.B, current_game_state.player2.player_buttons.X,
                current_game_state.player2.player_buttons.A, current_game_state.player2.player_buttons.L, current_game_state.player2.player_buttons.R,
                current_game_state.player2.player_buttons.select, current_game_state.player2.player_buttons.start])

    # Define header
    headers = [
        'timer', 'fight_result', 'has_round_started', 'is_round_over',
        'player1_id', 'player1_health', 'player1_x_coord', 'player1_y_coord', 'player1_jumping', 'player1_crouching', 'player1_in_move', 'player1_move_id',
        'player1_up', 'player1_down', 'player1_right', 'player1_left',
        'player2_id', 'player2_health', 'player2_x_coord', 'player2_y_coord', 'player2_jumping', 'player2_crouching', 'player2_in_move', 'player2_move_id',
        'player2_up', 'player2_down', 'player2_right', 'player2_left',
        'player1_Y', 'player1_B', 'player1_X', 'player1_A', 'player1_L', 'player1_R', 'player1_select', 'player1_start',
        'player2_Y', 'player2_B', 'player2_X', 'player2_A', 'player2_L', 'player2_R', 'player2_select', 'player2_start'
    ]

    file_exists = os.path.isfile('game_log.csv')
    is_empty = not file_exists or os.stat('game_log.csv').st_size == 0

    with open('game_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if is_empty:
            writer.writerow(headers)
        writer.writerow(row)


def main():
    if (sys.argv[1]=='1'):
        client_socket = connect(9999)
    elif (sys.argv[1]=='2'):
        client_socket = connect(10000)
    current_game_state = None
    #controller = KeyboardController()  # <-- Human player

    #print( current_game_state.is_round_over )
    #bot=Bot()
    
    # Load scaler from processed_data.pkl
    with open('processed_data.pkl', 'rb') as f:
        _, _, _, _, scaler = pickle.load(f)
    
    # Initialize ML Bot
    controller = MLBot(
        model_path='trained_model.h5',
        scaler_path='processed_data.pkl'  # Or 'scaler.pkl' if separate
    )
    print("ML Bot ready!")
    bot = MLBot()
    while (current_game_state is None) or (not current_game_state.is_round_over):
        current_game_state = receive(client_socket)
        #writeGameState(current_game_state)
        print("Row written to CSV!")
        # bot_command = bot.fight(current_game_state, sys.argv[1])
        # send(client_socket, bot_command)
        # game_command = controller.get_command(current_game_state, sys.argv[1])
        # send(client_socket, game_command)
        game_command = controller.predict_action(current_game_state)
        send(client_socket, game_command)
    


# Add at top


# Modify main()
# def main():
#     client_socket = connect(9999 if sys.argv[1] == '1' else 10000)
    
#     # Add at the top

#     logging.basicConfig(level=logging.INFO)

#     # In main(), modify Player 1 initialization:
#     if sys.argv[1] == '1':
#         try:
#             # Load scaler from processed_data.pkl
#             with open('processed_data.pkl', 'rb') as f:
#                 _, _, _, _, scaler = pickle.load(f)
            
#             # Initialize ML Bot
#             controller = MLBot(
#                 model_path='trained_model.h5',
#                 scaler_path='processed_data.pkl'  # Or 'scaler.pkl' if separate
#             )
#             print("ML Bot ready!")
#         except Exception as e:
#             print(f"ML Bot failed to load: {str(e)}")
#             sys.exit(1)

#     current_game_state = None
#     while current_game_state is None or not current_game_state.is_round_over:
#         current_game_state = receive(client_socket)
#         #writeGameState(current_game_state)
        
#         game_command = controller.predict_action(current_game_state)
#         send(client_socket, game_command)
        
#         time.sleep(0.01)  # 60 FPS


if __name__ == '__main__':
    main()
   
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib

def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    def encode_action(row):
        is_enemy_attacking = row['player2_in_move'] and (
            row['player2_Y'] or row['player2_B'] or row['player2_X'] or row['player2_A'])
        distance = abs(row['player1_x_coord'] - row['player2_x_coord'])
        health_diff = row['player1_health'] - row['player2_health']
        opponent_health = row['player2_health']

        # Simulated conditions for special moves
        if distance >= 40 and row['player1_down'] and row['player1_left'] and row['player1_Y']:
            return 0  # Fireball
        elif distance < 70 and row['player1_up'] and row['player1_right'] and row['player1_B']:
            return 1  # Spinning Attack

        # Regular action logic
        elif distance < 60 and row['player1_B']:  # Heavy attack
            return 2
        elif distance < 60 and row['player1_Y']:  # Light attack
            return 6
        elif opponent_health < 30 and row['player1_B']:  # Heavy attack to finish
            return 2
        elif is_enemy_attacking and distance < 30 and row['player1_down']:  # Crouch block
            return 1
        elif distance > 70 and row['player1_right']:  # Advance
            return 3
        elif distance < 15 and row['player1_left']:  # Retreat
            return 4
        elif is_enemy_attacking and distance < 15 and row['player1_up']:  # Jump to dodge
            return 5
        else:
            return 7  # Idle fallback

    df['action'] = df.apply(encode_action, axis=1)
    df = df[df['action'] != -1]

    # Oversample aggressive/special actions
    attack_rows = df[df['action'].isin([0, 1, 2, 6])]
    df = pd.concat([df, attack_rows.sample(frac=0.7, replace=True)])

    feature_columns = [
        'timer', 'has_round_started', 'is_round_over',
        'player1_health', 'player1_x_coord', 'player1_y_coord',
        'player2_health', 'player2_x_coord', 'player2_y_coord',
        'player1_jumping', 'player1_crouching', 'player1_in_move', 'player1_up',
        'player1_down', 'player1_right', 'player1_left',
        'player2_jumping', 'player2_crouching', 'player2_in_move',
        'player2_up', 'player2_down', 'player2_right', 'player2_left',
        'player1_Y', 'player1_B', 'player1_X', 'player1_A',
        'player2_Y', 'player2_B', 'player2_X', 'player2_A',
        'distance', 'health_diff'
    ]

    X = df[feature_columns].values
    y = df['action'].values
    return X, y

# Load and preprocess data
X, y = load_data_from_csv("processed_fight_data.csv")

# Show action distribution
action_counts = pd.Series(y).value_counts()
print("Action distribution:", action_counts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1  # Avoid division by zero
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(8, activation='softmax'))  # Output layer for 8 action classes

optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)]

# Train the model
model.fit(X_train, y_train,
          epochs=100,
          batch_size=64,
          validation_split=0.2,
          callbacks=callbacks,
          class_weight=class_weight_dict,
          verbose=1)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save model and preprocessing stats
model.save('fighting_game_model.keras')
joblib.dump({'mean': mean, 'std': std}, 'preprocessing_stats.pkl')
print("Model and preprocessing stats saved.")

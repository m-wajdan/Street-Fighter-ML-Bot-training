# Street-Fighter-ML-Bot-training
A machine learning bot trained to play Street Fighter using reinforcement learning


## Pre Requisites 
- Make sure you have python version 3.11
- install pynput -> pip install pynput
- install tensor flow -> pip install tensorflow
- install numpy -> pip install numpy
- install pandas -> pip install pandas
- install sklearn -> pip install scikit-learn
- install joblib -> pip install joblib

### Running the game and executing the API code
1. For running a single bot (your bot vs CPU), open the single-player folder.
2. Run EmuHawk.exe.
3. From File drop down, choose Open ROM. (Shortcut: Ctrl+O)
4. From the same single-player folder, choose the Street Fighter II Turbo (U).smc file.
5. From Tools drop down, open the Tool Box. (Shortcut: Shift+T)
6. Once you have performed the above steps, leave the emulator window and tool box open and open the command prompt in the directory of the API and run the following commands:

    For Python API: `python controller.py 1`
    Note: The '1' at the end of each execution command is a command-line argument. '1' is for controlling player 1 through your bot (left hand side player in the game). Any command-line arguments other than '1' (without the quotes) will cause the code to give an error. (Yes we haven't done exception handling. Deal with it.)

7. After executing the code, go and select your character(s) in the game after choosing normal mode. 
8. Now click on the second icon in the top row (Gyroscope Bot). This will cause the emulator to establish a connection with the program you ran and you will see "Connected to the game!" on the terminal.
9. If you have completed all of these steps successfully then you have successfully run the Street FIghter Final Trained Model.
10. The program will stop once a single round is finished. Repeat this process for next Round

### Additional Modifications 
## HUMAN VS CPU
- For human vs CPU we added additional py module named as keyboard_controller.py, where we imported a library pynput ( which is better and lightweight than other lib like pygame)
- Using the pynput lib, we used the keyboard listner and then based on the press and release of the buttons we triggered the values of the buttons object in the command object.
- Whenever we create the new object keyboard controller, the constructor of this class makes a connection between the specific keys and the attributes. Then, on real time, the values of the attributes change on the basis of press and release.
- In the game loop, get comand func() from keyboardController class returns the command object with the updated values of its buttons.

## SAVE DATA IN CSV:
- Add a function in controller.py file, that recieves gamestate as an argument from the main games while loop 
- We are taking all the attributes of both players using the game state object
- The data is saved in csv which is further normalized and used for Model Training

#### Additional Files and Libraries :
File  # 1:  Trainer.py
Libraries + Imports:  
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from sklearn.model_selection import train_test_split
            from sklearn.utils.class_weight import compute_class_weight
            import joblib
Implementation:

File # 2:   ml_bot.py
Libraries + Imports:  
            import tensorflow as tf
            import joblib
            import numpy as np
            from command import Command
            from buttons import Buttons
Implementation:

### Alteration in Controller.py :
Libraries + Imports:
            import numpy as np
            import time
            from ml_bot import MLBot
            from command import Command

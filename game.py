import cv2
import numpy as np
from random import choice
from keras.models import load_model

# Trained Model
TRAINED_MODEL = "rock-paper-scissors-trained.h5"

# Defining class map for each label
CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
}


def mapper(val):
    return CLASS_MAP[val]


# Returns the correct winner based on the inputs received.
def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "Naman"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "Naman"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "Naman"
        if move2 == "rock":
            return "Computer"


# Loading the trained model
model = load_model(TRAINED_MODEL)

# Starting the video capture
cap = cv2.VideoCapture(0)
prev_move = None

# Drawing 2 Rectangle boxes in the camera frame
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # rectangle for computer to play
    cv2.rectangle(frame, (400, 50), (600, 250), (255, 255, 255), 2)
    # rectangle for user to play
    cv2.rectangle(frame, (50, 50), (350, 350), (255, 255, 255), 2)

    # specifying to only detect the hand inside the rectangle
    roi = frame[50:350, 50:350]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (155, 155))

    # predicting the move
    pred = model.predict(np.array([img]))
    print(pred[0])
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # deciding the winner
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
    prev_move = user_move_name

    # Text & Formatting
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Naman's Move: " + user_move_name,
                (25, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (325, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (150, 410), font, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

    # displaying the computer's move
    if computer_move_name != "none":
        icon = cv2.imread(
            "computerImages/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (200, 200))
        frame[50:250, 400:600] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    # Killing the game if the "Q" key is pressed
    stopKey = cv2.waitKey(10)
    if stopKey == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

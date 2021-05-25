import cv2
import numpy as np
from random import choice
from keras.models import load_model

# Trained Model
TRAINED_MODEL = "rock-paper-scissors-trained.h5"

# Defining class map for each label
CLASS_MAP = {
    0: "Rock",
    1: "Paper",
    2: "Scissors",
}


def mapper(val):
    return CLASS_MAP[val]


# Returns the correct winner based on the inputs received.
def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "Rock":
        if move2 == "Scissors":
            return "Naman"
        if move2 == "Paper":
            return "Computer"

    if move1 == "Paper":
        if move2 == "Rock":
            return "Naman"
        if move2 == "Scissors":
            return "Computer"

    if move1 == "Scissors":
        if move2 == "Paper":
            return "Naman"
        if move2 == "Rock":
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
    cv2.rectangle(frame, (370, 85), (620, 335), (255, 255, 255), 2)
    # rectangle for user to play
    cv2.rectangle(frame, (30, 70), (350, 350), (255, 255, 255), 2)

    # specifying to only detect the hand inside the rectangle
    roi = frame[70:350, 30:350]
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
            computer_move_name = choice(['Rock', 'Paper', 'Scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
    prev_move = user_move_name

    # Text & Formatting
    font_winner = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_move = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "NAMAN'S MOVE:",
                (87, 35), font_move, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, user_move_name,
                (157, 60), font_move, 0.7, (0, 155, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "COMPUTER'S MOVE:",
                (365, 35), font_move, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, computer_move_name,
                (450, 60), font_move, 0.7, (0, 155, 255), 2, cv2.LINE_AA)
    if winner != "Tie":
        cv2.putText(frame, winner + " won the round!",
                (30, 420), font_winner, 1.8, (0, 255, 0), 4, cv2.LINE_AA)
    else:
        cv2.putText(frame, "It's a " + winner + "!",
                (210, 420), font_winner, 2, (0, 255, 0), 4, cv2.LINE_AA)

    # displaying the computer's move
    if computer_move_name != "none":
        icon = cv2.imread(
            "computerImages/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (250, 250))
        frame[85:335, 370:620] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    # Killing the game if the "Q" key is pressed
    stopKey = cv2.waitKey(10)
    if stopKey == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

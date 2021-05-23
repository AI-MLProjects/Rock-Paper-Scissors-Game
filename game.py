from keras.models import load_model
import cv2
import numpy as np
from random import choice

CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
}


def mapper(val):
    return CLASS_MAP[val]


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


model = load_model("rock-paper-scissors-trained.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    cv2.resize(frame, (1280, 720))
    if not ret:
        continue
    # rectangle for computer to play
    cv2.rectangle(frame, (400, 50), (600, 250), (255, 255, 255), 2)
    # rectangle for user to play
    cv2.rectangle(frame, (50, 50), (350, 350), (255, 255, 255), 2)


    # extract the region of image within the user rectangle
    roi = frame[50:350, 50:350]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (155, 155))

    # predict the move made
    pred = model.predict(np.array([img]))
    print(pred[0])
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['rock', 'paper', 'scissors'])
            winner = calculate_winner(user_move_name, computer_move_name)
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Naman's Move: " + user_move_name,
                (25, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (325, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (150, 410), font, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (200, 200))
        frame[50:250, 400:600] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
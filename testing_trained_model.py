from keras.models import load_model
import cv2
import numpy as np

# Path where the sample image is stored
FILE_PATH = "test_img.png"

# Path where the trained model is stored
MODEL_PATH = "rock-paper-scissors-trained.h5"

# Defining class map for each label
CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
}
def mapper(val):
    return CLASS_MAP[val]

# Loading the trained model
model = load_model(MODEL_PATH)

# Preparing the image
img = cv2.imread(FILE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (148, 148))

# Predicting the name
pred = model.predict(np.array([img]))
pred_label = np.argmax(pred[0])
label_value = mapper(pred_label)

# Predicting the final result
print("Predicted: {}".format(label_value))

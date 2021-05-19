import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

DATASET_NAME = 'rock_paper_scissors'

# Importing & Extracting the RPS dataset.
(dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
    name=DATASET_NAME,
    data_dir='temp_datasets',
    with_info=True,
    as_supervised=True,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
)

get_label_name = dataset_info.features['label'].int2str


# Exploring raw training dataset images.
def preview_dataset(dataset):
    plt.figure(figsize=(12, 12))
    plot_index = 0
    for features in dataset.take(12):
        (image, label) = features
        plot_index += 1
        plt.subplot(3, 4, plot_index)
        label = get_label_name(label.numpy())
        plt.title('Label: %s' % label)
        plt.imshow(image.numpy())
        plt.show()

preview_dataset(dataset_train_raw)




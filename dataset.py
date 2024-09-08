from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def preprocess_data(x,y):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 画像を正規化（0〜255から0〜1へ）
    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    # One-hotエンコーディング
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images[:x], train_labels[:x], test_images[:y], test_labels[:y]
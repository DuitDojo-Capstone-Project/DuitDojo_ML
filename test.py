import tensorflow as tf
from main import load_and_preprocess_data

MODEL_PATH = 'capstone.h5'

model = tf.keras.models.load_model(MODEL_PATH)
trainImage, trainTargets, trainFileNames = load_and_preprocess_data('data', 'train')
testImage, testTargets, testFileNames = load_and_preprocess_data('data', 'test')
print(testImage[0].shape)
model.summary()
# Evaluate the model on the test dataset
train = model.evaluate(trainImage, trainTargets)
results = model.evaluate(testImage, testTargets)

# Print the evaluation results
print("train Loss:", train[0])
print("train Accuracy:", train[1])
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])
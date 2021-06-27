# Import libraries
import numpy as np # for processing of arrays
import pandas as pd
import sklearn # to display model performance on test set
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt # to display images from dataset
import os
from glob import glob
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model

# import tensorflow backend and keras api
import tensorflow as tf
import keras
import keras.backend as K

# import model layers and InceptionV3 architecture
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.ensemble import RandomForestClassifier

# import optimizers and callbacks
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint

from tensorflow.python.keras.layers import VersionAwareLayers

layers = VersionAwareLayers()
Dropout = layers.Dropout
Dense = layers.Dense
Input = layers.Input
concatenate = layers.concatenate
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
AveragePooling2D = layers.AveragePooling2D
Flatten = layers.Flatten

#Check if CUDA is available
print("GPU:", tf.config.list_physical_devices('GPU'), "\nCUDA Enabled:", tf.test.is_built_with_cuda(), "\nGPU Name:", tf.test.gpu_device_name(), "\nVisible Devices:", tf.config.experimental.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Download the data sets
#!wget -nv -q --show-progress -O ff.zip https://bit.ly/3w0xyBl
#!unzip -q ff.zip -d ffdata
#!rm ff.zip

# ImageDataGenerator loads images into memory in batches of specified size (in this case 16 images per batch)
# this avoids possible memory issues
train_folder = '/content/ffdata/train'
val_folder = '/content/ffdata/val'
df_train = pd.read_csv(train_folder + '/image_labels.csv')
df_val = pd.read_csv(val_folder + '/image_labels.csv')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # rescaling normalizes pixel values from the range [0,255] to [0,1]
train_set = datagen.flow_from_dataframe(dataframe=df_train, directory=train_folder, classes= ['real', 'fake'], class_mode="categorical", target_size=(299, 299), batch_size=16)
val_set = datagen.flow_from_dataframe(dataframe=df_val, directory=val_folder, classes= ['real', 'fake'], class_mode="categorical", target_size=(299, 299), batch_size=16)

print("Check class name mapping to label index:")
print(train_set.class_indices)
print(val_set.class_indices)

#Load InceptionV3 Pretrained Model
base_model = InceptionV3(include_top=False, weights="imagenet", input_tensor=Input(shape=(299,299,3)))
# add global pooling and dense layers to obtain output from the model
layer = base_model
layer = GlobalAveragePooling2D()(layer.output)
layer = Dense(2, activation='softmax', name='output')(layer)
input_layer = base_model.input
model = Model(inputs=input_layer, outputs=layer, name="InceptionV3")

# Optimisers from Keras https://keras.io/api/optimizers/#available-optimizers
sgd = SGD(lr=0.001, momentum=0.9, nesterov=False)
adadelta = Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07)
rmsprop = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False)

# Compile model
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
#!mkdir "/content/weights/"
# ModelCheckpoint callback saves the model weights after every epoch (iteration through the dataset)
# if the validation accuracy is higher than that of the model previously saved
checkpoint = ModelCheckpoint("/content/weights/inceptionv3.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# Train the model
hist = model.fit(train_set, steps_per_epoch=16, epochs = 25, validation_data=val_set, validation_steps=8, callbacks = [checkpoint])

# plot training and validation accuracy against epochs using matplotlib
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# plot training and validation loss against epochs using matplotlib
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Load Weights
model.load_weights("/content/weights/inceptionv3.h5")

# Evaluate the model

def read_image_from_disk(path):
    
  """
  Helper function to read image from disk given a absolute path.

  :param path: Absolute path to image file on disk
  :return: Image in Numpy Ndarray representation
  """

  img = tf.keras.preprocessing.image.load_img(path, target_size=(299,299,3))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = img/255
  img = np.expand_dims(img, axis=0)

  return img


def get_frames_to_vid_mapping(frame_list):

  """
  Helper function to generate a mapping of frames to it's corresponding video 
  name.

  The path of frames in the frame_list will be in such format:
  image/[video name]/[frame number].jpg
  e.g. image/00000/00032.jpg

  :param frame_list: A list of paths to the image frames
  :return: A sorted dictionary with keys as the video name and value as the
           corresponding frames.
           e.g. of returned mapping dictionary:

            {
              "00000":[
                  "00032",
                  "00064",
                  .
                  .
                  .
                  "00487"
              ],
              "00001":[
                  "00000",
                  "00032",
                  .
                  .
                  .
                  "00392"
              ],
              .
              .
              .
              "00790":[
                  "00000",
                  "00027",
                  .
                  .
                  .
                  "00542"
              ]
            }
  """

  # Get all videos name
  vidnames = [frame.split("/")[1:2][0] for frame in frame_list]
  # Get only unique names
  vidnames = set(vidnames)
  # Init the mapping dict
  mapping = {vidname: [] for vidname in vidnames}

  # Add frames to to its corresponding list
  for frame in frame_list:
    vidname = str(frame.split("/")[1:2][0])
    frame_number = str(frame.split("/")[-1].split(".")[0])
    mapping[vidname].append(frame_number)

  return dict(sorted(mapping.items()))


def infer_videos(test_data_path, csv_file, num_of_videos='All'):

  """
  Function to infer a test data set. The function takes in a path to the test
  data set and a csv file that contain the paths of the frames extracted from 
  the videos in the test dataset.

  :param test_data_path: Absoulute path to the test dataset
  :param csv_file: File Name of the CSV file that must be in the test_data_path
  :param num_of_videos: Number of videos to infer from the dataset (default: All)
  :return: Pandas dataframe which contains the prediction (probability of being 
           fake) of each video. 
  """

  list_dir = list(pd.read_csv(test_data_path + csv_file).iloc[:,0])

  mapping = get_frames_to_vid_mapping(list_dir)

  # [*mapping] gives the list of keys (video name) in the mapping dict
  num_of_videos_avail = len([*mapping])

  # Set number of videos to be inferred to total of videos available if given 
  # num_of_videos is more than max amount of available videos
  if num_of_videos == 'All' or num_of_videos > num_of_videos_avail:
      num_of_videos = num_of_videos_avail

  # init mapping of videos to its corresponding predicted probabilities
  videos_to_prediction = {}

  # Loop through each video and make a prediction of each frame in the video.
  # Assigned a prediction to each video by taking the mean of its corresponding
  # frames' probabilities.
  for video_name in [*mapping][0:num_of_videos]:

    frames = mapping[video_name]
    predictions = []
    print("Infering video {video}...".format(video=video_name))
    print("Processing frame ", end=" ")

    # Process each frame in video
    for frame in frames:
      print(frame, end =", ")
      frame_path = "image/{video_name}/{frame}.jpg".format(video_name=video_name, frame=frame)
      img = read_image_from_disk(test_data_path + frame_path)
      prediction = model.predict(img)[0]
      # Collect only the 'real' side of probability
      predictions.append(prediction[1])

    # Take the mean of the probabilities from the frames
    videos_to_prediction[video_name] = statistics.mean(predictions)
    print("Done!")
  
  return pd.DataFrame(videos_to_prediction.items())

modelPredictions = infer_videos("/content/ffdata/test/", "image_labels.csv")
print(modelPredictions)

# Submission
modelPredictions.columns = ['vid_name', 'label']
modelPredictions.to_csv("/content/ffdata/model_predictions.csv", index=False)

# InceptionV3_DeepFakes_Model
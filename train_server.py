#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#GPU configs
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#convert the model generated to tflite
def convert_tflite(modelname,modelh5):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(modelh5)
        tflite_model = converter.convert()
        open(modelname+'.tflite', "wb").write(tflite_model)
        print("Saved TFLite model with name: ",modelname+'.tflite')

    except Exception as e:
        print("Something went wrong while converting to TFlite. Error: ",e)

# Build a Keras model given some parameters
def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
  input_tensor = keras.Input(input_shape)
  x = input_tensor
  for i, module_length in enumerate([module_size] * model_depth):
      for j in range(module_length):
          x = keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
          x = keras.layers.BatchNormalization()(x)
          x = keras.layers.Activation('relu')(x)
      x = keras.layers.MaxPooling2D(2)(x)

  x = keras.layers.Flatten()(x)
  x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d'%(i+1))(x) for i in range(captcha_length)]
  model = keras.Model(inputs=input_tensor, outputs=x)

  return model

# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 1), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in range(self.captcha_length)]

        for i in range(self.batch_size):
            try:
                random_image_label = random.choice(list(self.files.keys()))
                random_image_file = self.files[random_image_label]
            except:
                print("Ran out of files in the batch!")
                break
            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            #converting to greyscale
            grey_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
            processed_data = numpy.array(grey_data) / 255.0
            #adding a third dimension to convert to shape (64,128,1)
            processed_data = numpy.expand_dims(processed_data, axis=2)
            X[i] = processed_data

            # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
            # So the real label should have the "_num" stripped out.
            random_image_label = random_image_label.split('_')[0]

            #Here we are doing the most important step
            #I'm adding a padding character ',' to match the length given in the command
            #The model will be trained with this label, where it will determine the empty space as the padded character (HYPOTHESIS)
            while (len(random_image_label)<self.captcha_length):
                random_image_label = random_image_label+","
            # print(random_image_label)
            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument('--train-dataset', help='Where to look for the training image dataset', type=str)
    parser.add_argument('--validate-dataset', help='Where to look for the validation image dataset', type=str)
    parser.add_argument('--output-model-name', help='Where to save the trained model', type=str)
    parser.add_argument('--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--epochs', help='How many training epochs to run', type=int)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.batch_size is None:
        print("Please specify the training batch size")
        exit(1)

    if args.epochs is None:
        print("Please specify the number of training epochs to run")
        exit(1)

    if args.train_dataset is None:
        print("Please specify the path to the training data set")
        exit(1)

    if args.validate_dataset is None:
        print("Please specify the path to the validation data set")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()
        captcha_symbols = captcha_symbols+','
        print(captcha_symbols)

    #Finding the GPU if present and using it to train the model.
    device = '/device:CPU:0'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: # "GPU available!"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = '/device:GPU:0'
        print("Using GPU for training: ",device)
    # print(captcha_symbols)
    with tf.device(device):
        # with tf.device('/device:XLA_CPU:0'):
        print(f'training with {device}')
        #model bein created with the shape (64,128,1)
        model = create_model(args.length, len(captcha_symbols), (args.height, args.width, 1))

        if args.input_model is not None:
            model.load_weights(args.input_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        training_data = ImageSequence(args.train_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)
        validation_data = ImageSequence(args.validate_dataset, args.batch_size, args.length, captcha_symbols, args.width, args.height)

        #stopping if the model has not improved in 2 epochs
        callbacks = [keras.callbacks.EarlyStopping(patience=2),
                     # keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(args.output_model_name+'.h5', save_best_only=False)]

        # Save the model architecture to JSON
        with open(args.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            history = model.fit_generator(generator=training_data,
                                validation_data=validation_data,
                                epochs=args.epochs,
                                callbacks=callbacks,
                                use_multiprocessing=True)
        except Exception as e:
            print(e)
            print('Error caught, saving current weights as ' + args.output_model_name+'_resume.h5')
            model.save_weights(args.output_model_name+'_resume.h5')
            exit()
        

        #Converting the model to TFlite for Pi
        convert_tflite(args.output_model_name,model)
        # print(history.history.keys())
        plt.figure(figsize=(12, 4))
        
        # Plot and save the training history (loss and accuracy)
        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_char_1_accuracy'])
        plt.plot(history.history['val_char_2_accuracy'])
        plt.plot(history.history['val_char_3_accuracy'])
        plt.plot(history.history['val_char_4_accuracy'])
        plt.plot(history.history['val_char_5_accuracy'])
        plt.plot(history.history['val_char_6_accuracy'])
        plt.title('Validation Character Accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

        plt.tight_layout()

        # Save the plots to a file
        plt.savefig('training_plots.png')

    #printing the time it took to train the model
    print("--- %s seconds --- taken to train" % (time.time() - start_time))

        # Show the plots (optional)
        # plt.show()

if __name__ == '__main__':
    main()

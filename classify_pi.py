#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import numpy as np
import argparse
import tensorflow as tf
import keras
from PIL import Image
import time
import csv
def decode(characters, y):
    # print(characters)
    y = np.argmax(np.array(y), axis=2)[:,0]
    # print(y)
    # print(characters[43])
    return ''.join([characters[x] for x in y])

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--shortname', help='Header of the CSV file', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.shortname is None:
        print("Please specify the shortname")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    captcha_symbols = captcha_symbols+','
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    device = '/device:CPU:0'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: # "GPU available!"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = '/device:GPU:0'
    # print(captcha_symbols)

    with tf.device(device):
        # with open(args.output, 'w') as output_file:
        with open(args.output, 'w', encoding='UTF8', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow([args.shortname])
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])
            
            captchas_files = sorted(os.listdir(args.captcha_dir))
            for x in captchas_files:
                # load image and preprocess it
                raw_data = Image.open(os.path.join(args.captcha_dir, x))
                if raw_data.mode != 'RGB':
                    raw_data = raw_data.convert('RGB')
                image = np.array(raw_data)/255.0
                (h, w, c) = image.shape
                image = image.reshape([-1, h, w, c])
                prediction = model.predict(image)
                writer.writerow([x,decode(captcha_symbols, prediction).replace(',','')])
                print('Classified ' + x + " as "+decode(captcha_symbols, prediction).replace(',',''))
    print("--- %s seconds --- taken to classify" % (time.time() - start_time))

if __name__ == '__main__':
    main()

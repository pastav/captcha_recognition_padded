#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import numpy as np
import argparse
import tflite_runtime.interpreter as tflite
# import keras
from PIL import Image
import time
import csv
def decode(characters, y):
    # print(characters)
    y = np.argmax(np.array(y), axis=1)
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
        print("Please specify the model to use")
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

    with open(args.output, 'w', encoding='UTF8', newline='') as output_file:
        writer = csv.writer(output_file)
        #adding shortname to csv
        writer.writerow([args.shortname])
        # Load the TFLite model in TFLite Interpreter
        modelname= args.model_name+'.tflite'
        print(modelname)
        interpreter = tflite.Interpreter(args.model_name+'.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # print("input details: ",input_details)
        # print("output details: ",output_details)
        input_shape = input_details[0]['shape']
        print("Input shape to tflite model: ",input_shape)

        captchas_files = sorted(os.listdir(args.captcha_dir))
        for x in captchas_files:
            # load image and preprocess it
            raw_data = Image.open(os.path.join(args.captcha_dir, x))
            # if raw_data.mode != 'RGB':
            #     raw_data = raw_data.convert('RGB')
            #converting to greyscale
            gray_image = raw_data.convert('L')
            image = np.array(gray_image)/255.0
            image = np.expand_dims(image, axis=2)
            #convert to float 32 as pi is 32 bit system
            image = np.float32(image)
            (h, w, c) = image.shape
            image = image.reshape([-1, h, w, c])
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            #the tensor indexes were found out by comparing to the test dataset
            tensors_ind=[3,5,0,4,2,1]
            #getting the captcha label
            captcha_name=""
            for i in range(0, 6):
                captcha_name+=decode(captcha_symbols, interpreter.get_tensor(output_details[tensors_ind[i]]['index']))
            # output_data=decode(captcha_symbols, interpreter.get_tensor(output_details[0]['index']))
            # print(output_data)
            # print(captcha_name)
            #printing to csv
            writer.writerow([x,captcha_name.replace(',','')])
            print('Classified ' + x + " as "+captcha_name.replace(',',''))
    print("--- %s seconds --- taken to classify" % (time.time() - start_time))

if __name__ == '__main__':
    main()

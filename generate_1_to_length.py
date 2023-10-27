#!/usr/bin/env python3
import os
import numpy
import random
import cv2
import argparse
import captcha.image
import time
import matplotlib.pyplot as plt
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
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

    if args.length == 0:
        print("Please specify a non zero character length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)
    #initialize the captcha generator with the font file
    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=["EamonU.ttf"])

    #reading the symbol file to get the character set
    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        #creates the output directory if it does not exist
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
    n=1
    c=0
    timetaken,count = [],[]
    #this generates (number of captchas)*(length of captcha) number of captchas with length 1-length of captcha
    #Loop from 1 to the length of the captcha
    for k in range(1,(args.length+1)):
        for i in range(args.count):
            #getting random string from the character set of length k
            random_str = ''.join([random.choice(captcha_symbols) for j in range(k)])
            image_path = os.path.join(args.output_dir, random_str+'.png')
            if os.path.exists(image_path):
                version = 1
                while os.path.exists(os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')):
                    version += 1
                image_path = os.path.join(args.output_dir, random_str + '_' + str(version) + '.png')
            #generating the image from the random string
            image = numpy.array(captcha_generator.generate_image(random_str))
            cv2.imwrite(image_path, image)
            #printing the total progress of the generation
            if (n%100==0):
                timetaken.append(round(time.time(),2))
                count.append(n)
                print("Done with: {}%".format(round((n/(args.count*args.length))*100)))
                c+=1
                # print(c)
            # print(n)
            n+=1
    print("--- %s seconds --- taken to generate the captchas" % round((time.time() - start_time),2))
    plt.plot(timetaken, count)
    plt.xlabel('Time Taken')
    plt.ylabel('Generated Captcha Count')
    plt.title('Time taken in seconds')
    plt.savefig('generating_time.png')

if __name__ == '__main__':
    main()

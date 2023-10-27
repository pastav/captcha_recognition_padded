# captcha_recognition_padded

Warning: These python files will run on Linux with limited support for Windows.

If you're using windows, consider using WSL as the generated captchas will require escaping characters if you're using Windows.

1.  Get files from the file server:

    python3 getfiles.py

    This will get the csv from the flle system, and then loop over the rows and download each png file.
    It will retry every file if the server drops the request till the file is downloaded.

    Change the shortname inside the script.

3.  Generate the captchas:

    python3 generate_1_to_length.py --width 128 --height 64 --length 6 --symbols symbols.txt --count 20000 --output-dir training_data

    This will generate captchas from length 1-6 and count 20000 times 6 (number of captchas) * (length of captcha)
    It will also generate a plot giving the count of images generated vs time.

4.  Train the model:

    nohup python3 train_server.py --width 128 --height 64 --symbols symbols.txt --length 6 --batch-size 32 --epochs 15 --output-model best_greyscale --train-dataset training_data --validate-dataset validation_data > out.txt 2>&1 </dev/null &

    We will be using nohup to run the training in the background and store the command output in out.txt file

    Please use the below command to follow up on the output of the command.

    tail -f out.txt

    This script will give the model in h5 and tflite format. The conversion happens automatically.

6.  Classify the captchas:

    You can use classify_tflite.py or classify_h5.py to classify the images.

    python3 classify_h5.py --model-name best_bw --captcha-dir finalcapwithtime --output output_bw.csv --symbols symbols.txt --shortname srivastp

    python3 classify_tflite.py --model-name converted_model --captcha-dir finalcapwithtime --output out_tflite.csv --symbols symbols.txt --shortname srivastp

    shortname is required as it is printed in the header of the csv file.

    These scripts automatically saves a CSV file which is suitable for submission to submitty


Note: to escape the characters,
you can use mapping of the characters as follows:

mapping = {'{': 101, '}': 102, '#': 103, '%': 104, '+' :105, '-': 106, ':': 107, '[': 108, '\\': 109, ']': 110}

In Train_server.py:

newstr=random_image_label
for key, value in mapping.items():
    if str(value) in newstr:
        newstr = newstr.replace(str(value), key)

In Generate_1_to_length.py:

c = 0
for i in range(0,len(random_str)):
    c = 0
    for key, value in mapping.items():
        if random_str[i] == key:
            if(i==0):
                newstr = newstr+str(value)+','
            elif(i==len(random_str) or i==(len(random_str)-1)):
                newstr = newstr +','+str(value)
            else:
                newstr = newstr+','+str(value)+','
            c = 1
    if c == 0:
        newstr = newstr+random_str[i]

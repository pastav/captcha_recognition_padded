import wget
import pandas as pd
import time
start_time = time.time()

filenames= "https://cs7ns1.scss.tcd.ie/?shortname=srivastp"
locationnames= "/users/pgrad/srivastp/project2"
retries = 1
success = False
while not success:
    try:
        wget.download(filenames, out=locationnames)
        # print("downloaded=",row[0])
        success = True
        print("Success with the filenames csv")
    except Exception as e:
        print('Error while downloading filenames.csv retrying',retries)
        retries += 1

url = "https://cs7ns1.scss.tcd.ie"
shortname = "srivastp"
location= "/users/pgrad/srivastp/project2/finalcapwithtime"
df = pd.read_csv("srivastp-challenge-filenames.csv",header=None)

for index, row in df.iterrows():
    retries = 1
    success = False
    #print(row[0])
    entireurl = url+"/?shortname="+shortname+"&myfilename="+row[0]
    while not success:
        try:
            wget.download(entireurl, out=location)
            # print("downloaded=",row[0])
            success = True
            print("Success with ",row[0])
        except Exception as e:
            print('Error! retrying',retries)
            retries += 1

print("--- %s seconds ---" % round(time.time() - start_time, 2))
    




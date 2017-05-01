import urllib.request

# following script fetches images from "TIME PICKS THE TOP 100 PHOTOS OF THE YEAR" article
# http://time.com/top-100-photos-2016/
LOAD = 0

# Set LOAD to 1 to re-load images from the websites
if(LOAD):
    for i in range(1, 101):
        # link to file no. 59 is broken
        number = i
        if(i == 59):
            continue
        elif(i > 59):
            number = i - 1

        link = "https://timedotcom.files.wordpress.com/2016/12/100-photos-{}.jpg?quality=100".format(
            number)
        output_name = "photos-{}.jpg".format(number - 1)
        urllib.request.urlretrieve(link, output_name)
        print("Load {} successful.".format(output_name))


UPLOAD = 1
if(UPLOAD):
    

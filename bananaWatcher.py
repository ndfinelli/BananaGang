from sendEmail import *
from takePhoto import takePic

address2sendEmail2 = "ndfinelli@gmail.com"
subject = "Banana Watcher - output"
bananaResponses = ["Bananas are fairly green, you should wait a hot sec before consumption",
                   "Those nananers are right on the money, gobble them up b4 its too late",
                   "The bananas are about to turn, you only a day or two left b4 bad",
                   "You've got bad bananas :(    Guess we better make some banana bread"]


# objectDetection first
takePic()

# if objectDetection finds a banana run the pred through the model
"""
if(True):
  modPred = 0
  sendEmail( address2sendEmail2, subject, bananaResponses[modPred])
# else send email that you are out of nanas
else:
  sendEmail( address2sendEmail2, subject, "You are out of bananas bro... ")
"""
  

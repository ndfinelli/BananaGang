import smtplib

def sendEmail(address2send2, subject= "Are my bananas brown yet?", body=""):
  smtpUser = "brownbanana1234@gmail.com"
  smtpPass = "oveawleewhsvgzsg"

  # Address we want to send the email to
  toAdd = address2send2 
  fromAdd = smtpUser

  header = "To: " + toAdd + "\n" + "From: " + fromAdd + "\n" + "Subject: " + subject

  print (header + "\n" + body)

  s = smtplib.SMTP("smtp.gmail.com", 587)

  s.ehlo()
  s.starttls()
  s.ehlo()

  s.login(smtpUser, smtpPass)
  s.sendmail(fromAdd, toAdd, header + "\n\n" + body)

  s.quit()

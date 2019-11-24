Import smtplib

smtpUser = ‘areMyBananasbrown@gmail.com’
smtpPass = ‘banana1234!’

# Address we want to send the email to
toAdd = ‘address@gmail.com’ 
fromAdd = smtpUser

subject = ‘Are my bananas brown yet?’
header = ‘To: ‘ + toAdd + ‘\n’ + ‘From: ‘ + fromAdd + ‘\n’ + ‘Subject: ‘ + subject
body = ‘your bananas are good to go dude!’

print header + ‘\n’ + body

s = smtplib.SMTP(‘smtp.gmail.com’, 587)

s.ehlo()
s.startls()
S.ehlo()

s.login(smtpUser, smtpPass)
s.sendmail(fromAdd, toAdd, header + ‘\n\n’ + body)

s.quit()

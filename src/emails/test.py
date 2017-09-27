import smtplib
from sys import argv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

fromaddr = argv[1]
toaddr = argv[1]
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "BUY SIGNAL ALERT"

body = "YOUR MESSAGE HERE"
msg.attach(MIMEText(body, 'plain'))

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, argv[2])
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()

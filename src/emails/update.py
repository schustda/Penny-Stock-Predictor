import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sys import argv

def send_mail(username,password):

    gmailUser = ''
    gmailPassword = ''
    recipient = 'douglas.schuster303@gmail.com'
    message='your message here '

    msg = MIMEMultipart()
    msg['From'] = gmailUser
    msg['To'] = recipient
    msg['Subject'] = "Subject of the email"
    msg.attach(MIMEText(message))

    mailServer = smtplib.SMTP('smtp.gmail.com', 587)
    mailServer.ehlo()
    mailServer.starttls()
    mailServer.ehlo()
    print (gmailUser,gmailPassword)
    print (type(gmailUser),type(gmailPassword))
    mailServer.login(gmailUser, gmailPassword)
    mailServer.sendmail(gmailUser, recipient, msg.as_string())
    mailServer.close()


if __name__ == '__main__':
    send_mail(argv[1],argv[2])

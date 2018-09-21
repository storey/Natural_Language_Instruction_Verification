# send an email when report is done
import smtplib
from email.message import EmailMessage

# send an email that model name is done, along with the result report
def sendEmail(name, textReport):
    msg = EmailMessage()
    msg["Subject"] = "Model %s is done." % name
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg["From"] = "grantjstorey@gmail.com"
    msg["To"] = "grantjstorey@gmail.com"
    msg.set_content(textReport)

    # Send the email via our own SMTP server.
    s = smtplib.SMTP("localhost")
    s.send_message(msg)
    s.quit()

if __name__ == "__main__":
    sendEmail("Model1", "hahahaha")

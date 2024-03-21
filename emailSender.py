import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_attached_email(mail_subject, attach_file_path):
	mail_content = """
	No content, just audio
	"""
	#The mail addresses and password
	sender_address = 'l39X35f828DPWf9j@gmail.com'
	sender_pass = 'hizdnjdiyjabizja'
	receiver_address = "nuo_wen_lei@brown.edu"
	#Setup the MIME
	message = MIMEMultipart()
	message['From'] = sender_address
	message['To'] = receiver_address
	message['Subject'] = mail_subject
	#The subject line
	#The body and the attachments for the mail
	message.attach(MIMEText(mail_content, 'plain'))
	attach_file_name = 'audio.wav'
	attach_file = open(attach_file_path, 'rb') # Open the file as binary mode
	payload = MIMEBase('audio', 'wav')
	payload.set_payload((attach_file).read())
	encoders.encode_base64(payload) #encode the attachment
	#add payload header with filename
	payload.add_header('Content-Disposition', 'attachment', filename=attach_file_name)
	message.attach(payload)
	#Create SMTP session for sending the mail
	session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
	session.starttls() #enable security
	session.login(sender_address, sender_pass) #login with mail_id and password
	text = message.as_string()
	session.sendmail(sender_address, receiver_address, text)
	session.quit()

def send_email(body):
	# Set Global Variables
	gmail_user = 'l39X35f828DPWf9j@gmail.com'
	gmail_password = 'hizdnjdiyjabizja'
	to = "nuo_wen_lei@brown.edu"
	# Create Email 
	mail_from = f"stock_data_email <{gmail_user}>"
	mail_to = f"Nuo Wen Lei <{to}>"
	mail_subject = "CCV Run Status"
	mail_message_body = body

	mail_message = '''\
	From: %s\r\nTo: %s\r\nSubject: %s\r\nContent-Type: text/html\n%s
	''' % (mail_from, mail_to, mail_subject, mail_message_body)
	# Sent Email
	server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
	server.login(gmail_user, gmail_password)
	server.sendmail(gmail_user, to, mail_message)
	server.close()
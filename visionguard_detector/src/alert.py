import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import io
import cv2


def send_alert(frame):
    # Configurações do e-mail
    sender_email = "seu_email@example.com"
    receiver_email = "destinatario@example.com"
    subject = "Alerta: Objeto Cortante Detectado"
    body = "Um objeto cortante foi detectado na câmera de segurança."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))

    # Converte o frame em imagem e anexa ao e-mail
    img_byte_arr = io.BytesIO()
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_byte_arr.write(img_encoded)

    image_data = MIMEImage(img_byte_arr.getvalue())
    msg.attach(image_data)

    try:
        with smtplib.SMTP('smtp.example.com', 587) as server:  # Substitua pelo seu servidor SMTP
            server.starttls()
            server.login(sender_email, "sua_senha")  # Substitua pela sua senha
            server.send_message(msg)
            print("Alerta enviado com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar alerta: {e}")

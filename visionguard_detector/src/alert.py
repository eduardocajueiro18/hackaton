import base64
import cv2
import mailtrap as mt

# Configurações do Mailtrap API
TOKEN = "9d8ac5a72e61f1d6ed7df20190fde559"

# Configurações do e-mail
SENDER_EMAIL = "alert@demomailtrap.com"
SENDER_NAME = "Alerta de Segurança"
RECEIVER_EMAIL = "eduardocajueiro@gmail.com"
RECEIVER_NAME = "Eduardo Cajueiro"
SUBJECT = "Alerta: Objeto Cortante Detectado"
BODY = "Um objeto cortante foi detectado na câmera de segurança."

def send_alert(frame):
    # Codificar o frame para formato JPEG
    _, buffer = cv2.imencode(".jpg", frame)

    mail = mt.Mail(
        sender=mt.Address(email=SENDER_EMAIL, name=SENDER_NAME),
        to=[mt.Address(email=RECEIVER_EMAIL, name=RECEIVER_NAME)],
        subject=SUBJECT,
        text=BODY,
        category="Detecção de objeto cortante",
        attachments=[
            mt.Attachment(
                content=base64.b64encode(buffer.tobytes()),
                filename="frame.jpg",
                disposition=mt.Disposition.INLINE,
                mimetype="image/jpeg",
                content_id="frame.jpg",
            )
        ],
    )

    try:
        client = mt.MailtrapClient(token=TOKEN)
        client.send(mail)
        print("Alerta enviado com sucesso!")
    except mt.exceptions.MailtrapError as e:
        print(f"Erro ao enviar alerta: {e}")

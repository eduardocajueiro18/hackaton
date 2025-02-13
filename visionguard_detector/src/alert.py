# import base64
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage
# from email.mime.text import MIMEText
# import io
# import cv2
# import mt
#
# def send_alert(frame):
#     # Configurações do e-mail
#     sender_email = "seu_email@example.com"
#     receiver_email = "destinatario@example.com"
#     subject = "Alerta: Objeto Cortante Detectado"
#     body = "Um objeto cortante foi detectado na câmera de segurança."
#     token = "667ec16bb94d3a40ed66b7f772012494"
#
#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = subject
#
#     msg.attach(MIMEText(body, 'plain'))
#
#     # Converte o frame em imagem e anexa ao e-mail
#     img_byte_arr = io.BytesIO()
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     img_byte_arr.write(img_encoded)
#
#     image_data = MIMEImage(img_byte_arr.getvalue())
#     msg.attach(image_data)
#
#     # Converter o frame para JPEG
#     _, buffer = cv2.imencode(".jpg", frame)
#     image_data = buffer.tobytes()  # Converter para bytes
#
#     # Codificar para Base64
#     encoded_image = base64.b64encode(image_data)  # Retorna bytes diretamente
#
#     # Criar o anexo corretamente
#     # email_attachment = mt.Attachment(
#     #     content=encoded_image,  # Agora está correto, pois Base64 já retorna bytes
#     #     filename="frame.jpg",
#     #     disposition=mt.Disposition.INLINE,
#     #     mimetype="image/jpeg",
#     #     content_id="frame.jpg",
#     # )
#
#     try:
#         with smtplib.SMTP('smtp.example.com', 587) as server:  # Substitua pelo seu servidor SMTP
#             server.starttls()
#             server.login(sender_email, "sua_senha")  # Substitua pela sua senha
#             server.send_message(msg)
#             print("Alerta enviado com sucesso!")
#     except Exception as e:
#         print(f"Erro ao enviar alerta: {e}")

import base64
import requests
import cv2

# Configurações do Mailtrap API
MAILTRAP_API_URL = "https://send.api.mailtrap.io/api/send"
MAILTRAP_TOKEN = "667ec16bb94d3a40ed66b7f772012494"

# Configurações do e-mail
SENDER_EMAIL = "seu_email@example.com"
RECEIVER_EMAIL = "eduardocajueiro@gmail.com"
SUBJECT = "Alerta: Objeto Cortante Detectado"
BODY = "Um objeto cortante foi detectado na câmera de segurança."

def send_alert(frame):
    # Converter o frame para imagem JPEG
    _, buffer = cv2.imencode(".jpg", frame)
    image_data = buffer.tobytes()  # Converter para bytes

    # Codificar a imagem em Base64
    encoded_image = base64.b64encode(image_data)  # Converte para string Base64

    # Criar o payload do e-mail
    payload = {
        "from": {"email": SENDER_EMAIL, "name": "Alerta de Segurança"},
        "to": [{"email": RECEIVER_EMAIL}],
        "subject": SUBJECT,
        "text": BODY,  # Corpo do e-mail em texto puro
        "attachments": [
            {
                "content": encoded_image,  # Imagem codificada em Base64
                "filename": "frame.jpg",
                "type": "image/jpeg",
                "disposition": "inline",
                "content_id": "frame.jpg"
            }
        ]
    }

    # Enviar a requisição HTTP para Mailtrap
    headers = {
        "Authorization": f"Bearer {MAILTRAP_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(MAILTRAP_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Lança erro se a requisição falhar
        print("Alerta enviado com sucesso!")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao enviar alerta: {e}")

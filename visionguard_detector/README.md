# FIAP VisionGuard - Detecção de Objetos Cortantes

## Descrição do Projeto

Este projeto tem como objetivo desenvolver um sistema de detecção de objetos cortantes (facas, tesouras e similares) em vídeos de segurança, utilizando Inteligência Artificial. O sistema emite alertas (via e-mail) para a central de segurança quando um objeto perigoso é detectado, auxiliando na prevenção de incidentes.

Este projeto foi desenvolvido como parte do Hackathon da FIAP, com o objetivo de validar a viabilidade da funcionalidade de detecção supervisionada de objetos cortantes.

## Funcionalidades

*   **Detecção de Objetos Cortantes:** Identificação em tempo real de facas, tesouras e objetos similares em vídeos.
*   **Sistema de Alertas:** Envio de e-mails com imagens dos objetos detectados para a central de segurança.
*   **Modelo Supervisionado:** Utilização de um modelo de aprendizado supervisionado treinado com um dataset customizado.

## Demonstração

[Link para o vídeo de demonstração no YouTube/Vimeo](Substitua por aqui o link para o seu vídeo)

## Como Começar

Siga estas instruções para configurar e executar o projeto no seu ambiente.

### Pré-requisitos

*   Python 3.7+
*   Pip (gerenciador de pacotes do Python)
*   [Opcional] Uma GPU para acelerar o treinamento e a inferência do modelo

### Instalação

1.  Clone o repositório:

    ```
    git clone [URL do seu repositório]
    cd visionguard_detector
    ```

2.  Crie um ambiente virtual (recomendado):

    ```
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    venv\Scripts\activate.bat  # No Windows
    ```

3.  Instale as dependências:

    ```
    pip install -r requirements.txt
    ```

4.  Configurar as variáveis de ambiente:
    * Crie um arquivo `.env` na raiz do projeto.
    * Adicione as seguintes variáveis:
        ```
        EMAIL_REMETENTE=seu_email@gmail.com
        SENHA_REMETENTE=sua_senha
        EMAIL_DESTINATARIO=email_da_central@example.com
        ```
    **Atenção:** Use um e-mail dedicado para este projeto e considere usar senhas de aplicativo (app passwords) se o seu provedor de e-mail exigir.

### Execução

1.  **Para executar a detecção em um vídeo:**

    ```
    python src/detector.py --video_path caminho/para/seu/video.mp4
    ```

    Substitua `caminho/para/seu/video.mp4` pelo caminho do vídeo que você deseja analisar.

### Treinamento do Modelo (Opcional)

1.  **Prepare o dataset:**
    *   Organize as imagens e anotações no formato correto (YOLO, COCO, etc.)
    *   Modifique o script `src/train.py` para carregar seu dataset.

2.  **Execute o treinamento:**

    ```
    python src/train.py
    ```

## Estrutura do Projeto
```
├── dataset                  # Arquivo principal para execução
├───── test/                 # Arquivo principal para execução
├───── train/                # Arquivo principal para execução
├── models/                  # Arquivo principal para execução
├── src/                     # Diretório do código-fonte
├───── alert.py              # Arquivo principal para execução
├───── detector.py           # Arquivo principal para execução
├───── train.py              # Arquivo principal para execução
├───── utils.py              # Arquivo principal para execução
├── videos/                  # Diretório de vídeos de teste
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação do projeto
```
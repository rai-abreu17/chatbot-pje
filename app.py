"""
app.py - Servidor Principal do Chatbot SeSJU
=============================================
Este é o ponto de entrada da aplicação. Ele cria o servidor Flask
que recebe mensagens tanto via API REST (para testes) quanto via
Webhook do Twilio (para WhatsApp).

Para iniciar o servidor:
    python app.py

Rotas disponíveis:
    POST /chat          → API REST para testes (JSON)
    POST /whatsapp      → Webhook do Twilio (WhatsApp)
    GET  /health        → Verificação de saúde do servidor
    POST /recarregar    → Recarrega a base de conhecimento
"""

import os
import tempfile
import requests
import speech_recognition as sr
from pydub import AudioSegment
from collections import defaultdict

from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse

from config import FLASK_PORT, FLASK_DEBUG, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
from rag_engine import RAGEngine


# ============================================================
# CACHE DE RESPOSTAS PENDENTES (para respostas longas)
# ============================================================
# Armazena a última resposta completa e partes restantes por usuário
respostas_pendentes = defaultdict(dict)

# Limite de caracteres por mensagem WhatsApp (seguro)
LIMITE_CARACTERES = 1500


def dividir_resposta(texto: str, limite: int = LIMITE_CARACTERES) -> list:
    """
    Divide uma resposta longa em partes menores, respeitando quebras de linha.
    """
    if len(texto) <= limite:
        return [texto]
    
    partes = []
    texto_restante = texto
    
    while texto_restante:
        if len(texto_restante) <= limite:
            partes.append(texto_restante)
            break
        
        # Tenta encontrar uma quebra de linha próxima do limite
        ponto_corte = texto_restante.rfind('\n', 0, limite)
        
        # Se não encontrar quebra de linha, tenta encontrar um ponto ou espaço
        if ponto_corte == -1 or ponto_corte < limite * 0.5:
            ponto_corte = texto_restante.rfind('. ', 0, limite)
        if ponto_corte == -1 or ponto_corte < limite * 0.5:
            ponto_corte = texto_restante.rfind(' ', 0, limite)
        if ponto_corte == -1:
            ponto_corte = limite
        
        partes.append(texto_restante[:ponto_corte + 1].strip())
        texto_restante = texto_restante[ponto_corte + 1:].strip()
    
    return partes


# ============================================================
# FUNÇÃO PARA TRANSCREVER ÁUDIO
# ============================================================
def transcrever_audio(media_url: str) -> str:
    """
    Baixa um arquivo de áudio do Twilio e transcreve para texto.
    
    Args:
        media_url: URL do arquivo de áudio no Twilio.
        
    Returns:
        Texto transcrito ou mensagem de erro.
    """
    try:
        print(f"[AUDIO] Baixando áudio de: {media_url}")
        
        # Baixa o áudio com autenticação do Twilio
        response = requests.get(
            media_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=30
        )
        response.raise_for_status()
        
        # Salva em arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_ogg:
            temp_ogg.write(response.content)
            temp_ogg_path = temp_ogg.name
        
        # Converte OGG para WAV (formato que o speech_recognition aceita)
        temp_wav_path = temp_ogg_path.replace(".ogg", ".wav")
        audio = AudioSegment.from_ogg(temp_ogg_path)
        audio.export(temp_wav_path, format="wav")
        
        # Transcreve usando Google Speech Recognition (gratuito)
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
            texto = recognizer.recognize_google(audio_data, language="pt-BR")
        
        print(f"[AUDIO] Transcrição: {texto}")
        
        # Limpa arquivos temporários
        os.unlink(temp_ogg_path)
        os.unlink(temp_wav_path)
        
        return texto
        
    except sr.UnknownValueError:
        print("[AUDIO] Não foi possível entender o áudio")
        return None
    except sr.RequestError as e:
        print(f"[AUDIO] Erro no serviço de transcrição: {e}")
        return None
    except Exception as e:
        print(f"[AUDIO] Erro ao processar áudio: {e}")
        return None

# ============================================================
# INICIALIZAÇÃO
# ============================================================
app = Flask(__name__)

# Inicializa o motor RAG (carrega documentos e cria índice)
print("=" * 60)
print("  CHATBOT ASSISTENTE SeSJU - Iniciando...")
print("=" * 60)
engine = RAGEngine()
print("=" * 60)
print("  SERVIDOR PRONTO!")
print("=" * 60)


# ============================================================
# ROTA 1: API REST PARA TESTES (sem WhatsApp)
# ============================================================
@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint para testes via API REST.
    
    Envie um POST com JSON:
        {"mensagem": "Como limpar o cache do Chrome?"}
    
    Retorna:
        {"resposta": "Para limpar o cache do Chrome..."}
    
    Teste com curl:
        curl -X POST http://localhost:5000/chat \
             -H "Content-Type: application/json" \
             -d '{"mensagem": "Como limpar o cache?"}'
    """
    dados = request.get_json()

    if not dados or "mensagem" not in dados:
        return jsonify({
            "erro": "Envie um JSON com o campo 'mensagem'.",
            "exemplo": {"mensagem": "Como faço para limpar o cache?"}
        }), 400

    pergunta = dados["mensagem"]
    print(f"\n[CHAT] Pergunta recebida: {pergunta}")

    resposta = engine.responder(pergunta)
    print(f"[CHAT] Resposta gerada: {resposta[:100]}...")

    return jsonify({"resposta": resposta})


# ============================================================
# ROTA 2: WEBHOOK DO TWILIO (WhatsApp)
# ============================================================
@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    """
    Endpoint que recebe mensagens do WhatsApp via Twilio.
    
    O Twilio envia os dados como form-data com os campos:
        - Body: texto da mensagem
        - From: número do remetente (formato: whatsapp:+55...)
        - To: número do bot
        - NumMedia: número de arquivos de mídia anexados
        - MediaUrl0: URL do primeiro arquivo de mídia
    
    Retorna TwiML (formato XML que o Twilio entende).
    """
    # Extrai dados da mensagem
    mensagem_recebida = request.form.get("Body", "").strip()
    remetente = request.form.get("From", "desconhecido")
    num_media = int(request.form.get("NumMedia", 0))

    print(f"\n[WHATSAPP] Mensagem de {remetente}: {mensagem_recebida}")
    
    # Verifica se é um áudio
    if num_media > 0:
        media_url = request.form.get("MediaUrl0", "")
        media_type = request.form.get("MediaContentType0", "")
        
        print(f"[WHATSAPP] Mídia recebida: {media_type}")
        
        if media_type.startswith("audio/"):
            # Transcreve o áudio
            texto_transcrito = transcrever_audio(media_url)
            
            if texto_transcrito:
                mensagem_recebida = texto_transcrito
                print(f"[WHATSAPP] Áudio transcrito: {mensagem_recebida}")
            else:
                resposta_texto = (
                    "Desculpe, não consegui entender o áudio. "
                    "Por favor, tente enviar uma mensagem de texto ou "
                    "grave o áudio novamente em um ambiente mais silencioso."
                )
                resp = MessagingResponse()
                resp.message(resposta_texto)
                return str(resp), 200, {"Content-Type": "application/xml"}
        else:
            # Outro tipo de mídia (imagem, vídeo, etc.)
            resposta_texto = (
                "Desculpe, no momento só consigo processar mensagens de texto "
                "e áudio. Por favor, envie sua dúvida por escrito."
            )
            resp = MessagingResponse()
            resp.message(resposta_texto)
            return str(resp), 200, {"Content-Type": "application/xml"}

    # Se a mensagem estiver vazia
    if not mensagem_recebida:
        resposta_texto = (
            "Olá! Sou o Assistente Virtual do SeSJU. "
            "Envie sua dúvida sobre o sistema que eu tentarei ajudar! "
            "Você pode enviar texto ou áudio."
        )
        resp = MessagingResponse()
        resp.message(resposta_texto)
        return str(resp), 200, {"Content-Type": "application/xml"}
    
    # Verifica se o usuário está pedindo continuação de resposta cortada
    msg_lower = mensagem_recebida.lower()
    pedindo_continuacao = any(palavra in msg_lower for palavra in [
        "cortada", "cortou", "continua", "continuação", "resto", 
        "completar", "termina", "terminar", "mais", "faltou"
    ])
    
    if pedindo_continuacao and remetente in respostas_pendentes and respostas_pendentes[remetente].get("partes"):
        # Envia a próxima parte da resposta anterior
        partes = respostas_pendentes[remetente]["partes"]
        proxima_parte = partes.pop(0)
        
        resp = MessagingResponse()
        if partes:
            resp.message(f"{proxima_parte}\n\n_(continua... envie 'mais' para ver o resto)_")
        else:
            resp.message(proxima_parte)
            del respostas_pendentes[remetente]
        
        print(f"[WHATSAPP] Enviando continuação ({len(partes)} partes restantes)")
        return str(resp), 200, {"Content-Type": "application/xml"}
    
    # Gera a resposta usando o motor RAG
    resposta_texto = engine.responder(mensagem_recebida)
    
    # Remove caracteres que podem quebrar o TwiML/XML
    resposta_texto = resposta_texto.replace('&', 'e').replace('<', '').replace('>', '')
    
    # Divide a resposta se for muito longa
    partes = dividir_resposta(resposta_texto)
    
    print(f"[WHATSAPP] Resposta completa ({len(resposta_texto)} chars):")
    print(f"[WHATSAPP] ---")
    print(resposta_texto)
    print(f"[WHATSAPP] ---")
    print(f"[WHATSAPP] Resposta dividida em {len(partes)} parte(s)")

    # Monta a resposta no formato TwiML (que o Twilio entende)
    resp = MessagingResponse()
    
    if len(partes) == 1:
        resp.message(partes[0])
    else:
        # Envia a primeira parte e guarda as outras
        resp.message(f"{partes[0]}\n\n_(continua... envie 'mais' para ver o resto)_")
        respostas_pendentes[remetente] = {"partes": partes[1:]}

    # Log do TwiML para debug
    twiml_str = str(resp)
    print(f"[WHATSAPP] TwiML completo ({len(twiml_str)} bytes):")
    print(twiml_str)
    
    return twiml_str, 200, {"Content-Type": "application/xml"}


# ============================================================
# ROTA 3: HEALTH CHECK (verificação de saúde)
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint para verificar se o servidor está funcionando.
    Útil para monitoramento e load balancers.
    """
    return jsonify({
        "status": "online",
        "servico": "Chatbot Assistente SeSJU",
        "documentos_indexados": engine.collection.count(),
    })


# ============================================================
# ROTA 4: RECARREGAR BASE DE CONHECIMENTO
# ============================================================
@app.route("/recarregar", methods=["POST"])
def recarregar():
    """
    Recarrega a base de conhecimento sem reiniciar o servidor.
    Útil quando o arquivo conhecimento.txt é atualizado.
    
    Teste com curl:
        curl -X POST http://localhost:5000/recarregar
    """
    try:
        engine.recarregar_documentos()
        return jsonify({
            "status": "sucesso",
            "mensagem": "Base de conhecimento recarregada.",
            "documentos_indexados": engine.collection.count(),
        })
    except Exception as e:
        return jsonify({
            "status": "erro",
            "mensagem": str(e),
        }), 500


# ============================================================
# ROTA 5: PÁGINA INICIAL (informativa)
# ============================================================
@app.route("/", methods=["GET"])
def index():
    """Página inicial com informações sobre o chatbot."""
    return jsonify({
        "nome": "Chatbot Assistente SeSJU",
        "versao": "1.0.0",
        "descricao": "Assistente virtual para suporte ao sistema SeSJU do TRT",
        "rotas": {
            "POST /chat": "API REST para testes (envie JSON com campo 'mensagem')",
            "POST /whatsapp": "Webhook do Twilio para WhatsApp",
            "GET /health": "Verificação de saúde do servidor",
            "POST /recarregar": "Recarrega a base de conhecimento",
        },
    })


# ============================================================
# INICIAR O SERVIDOR
# ============================================================
if __name__ == "__main__":
    print(f"\n>>> Servidor rodando em http://localhost:{FLASK_PORT}")
    print(f">>> Para testar: curl -X POST http://localhost:{FLASK_PORT}/chat "
          f'-H "Content-Type: application/json" '
          f'-d \'{{"mensagem": "Como limpar o cache?"}}\'')
    print(f">>> Para WhatsApp: configure o webhook do Twilio para /whatsapp\n")

    app.run(
        host="0.0.0.0",  # Aceita conexões de qualquer IP
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )

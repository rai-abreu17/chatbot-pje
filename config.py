"""
config.py - Configurações centralizadas do Chatbot SeSJU
=========================================================
Este arquivo carrega as variáveis de ambiente e define as
configurações globais do projeto. Para alterar qualquer
configuração, edite o arquivo .env na raiz do projeto.
"""

import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# ============================================================
# CONFIGURAÇÕES DO GOOGLE GEMINI
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Modelo recomendado para free tier: gemini-1.5-flash-latest (mais generoso em limites)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")

# ============================================================
# CONFIGURAÇÕES DO TWILIO (WhatsApp)
# ============================================================
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

# ============================================================
# CONFIGURAÇÕES DO SERVIDOR FLASK
# ============================================================
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# ============================================================
# CONFIGURAÇÕES DO RAG (Retrieval-Augmented Generation)
# ============================================================
# Caminho para o arquivo de conhecimento base
KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "conhecimento.txt")

# Tamanho dos "pedaços" de texto para indexação (em caracteres)
CHUNK_SIZE = 1000

# Sobreposição entre pedaços (para manter contexto entre eles)
CHUNK_OVERLAP = 250

# Número de trechos relevantes a recuperar por pergunta
TOP_K_RESULTS = 5

# ============================================================
# PROMPT DO SISTEMA (INSTRUÇÕES PARA A IA)
# ============================================================
# >>> EDITE AQUI PARA PERSONALIZAR O COMPORTAMENTO DO BOT <<<
# Este é o "cérebro" do assistente. Altere conforme necessário.
SYSTEM_PROMPT = """
Você é o Assistente PJe da Justiça Eleitoral. Seja simpático e prestativo.

FORMATAÇÃO:
- NÃO use markdown (nada de ** ou * para negrito)
- Texto simples, listas com números ou hífens
- Respostas adequadas para WhatsApp

COMO RESPONDER A PROBLEMAS:
Quando o usuário relatar lentidão, travamento, erro ou problema:
1. Explique a causa provável de forma simples
2. Pergunte se quer ver o passo a passo da solução

Exemplo de resposta CORRETA para "sistema lento":
"Isso geralmente acontece por causa de dados antigos salvos no cache do navegador. A limpeza do cache costuma resolver esse problema.

Quer que eu te mostre como fazer a limpeza no Chrome ou Firefox?"

COMO RESPONDER A SAUDAÇÕES:
Se o usuário mandar "oi", "olá", "bom dia":
"Olá! Sou o Assistente PJe da Justiça Eleitoral.

Posso te ajudar com:
- Problemas de lentidão ou travamento
- Dificuldades de acesso ou login  
- Recuperação de senha
- Uso do certificado digital

O que você precisa?"

REGRAS:
- Use APENAS as informações dos trechos abaixo
- Se não encontrar: "Não encontrei essa informação. Entre em contato com o suporte do seu tribunal."
- NÃO invente
- Sempre finalize de forma acolhedora

TRECHOS DA DOCUMENTAÇÃO:
---
{contexto}
---

Agora responda à pergunta do usuário seguindo as instruções acima.
"""

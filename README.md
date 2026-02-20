# Chatbot Assistente PJe - Justiça Eleitoral

**Versão 2.0.0**

Chatbot assistente integrado ao WhatsApp para responder dúvidas sobre o sistema PJe (Processo Judicial Eletrônico) da Justiça Eleitoral. Utiliza **Google Gemini** (gratuito) e arquitetura **RAG** (Retrieval-Augmented Generation) com ChromaDB.

---

## Funcionalidades

- Respostas baseadas na documentação oficial (arquivo `conhecimento.txt`)
- Integração com WhatsApp via Twilio
- Suporte a mensagens de áudio (transcrição automática)
- Busca híbrida: semântica + palavras-chave
- Expansão de sinônimos para linguagem coloquial
- Retry automático em caso de erro da API

---

## Arquitetura

| Componente | Tecnologia | Função |
|------------|------------|--------|
| Servidor Web | Flask | Webhook do WhatsApp + API REST |
| Motor RAG | ChromaDB + LangChain | Busca semântica na base de conhecimento |
| IA | Google Gemini | Geração de respostas |
| WhatsApp | Twilio | Integração com usuários |
| Transcrição | SpeechRecognition | Áudio para texto |

---

## Instalação

### 1. Pré-requisitos

- Python 3.10+
- FFmpeg (para áudio): `winget install ffmpeg` (Windows)
- Conta Google AI Studio (gratuita)
- Conta Twilio (gratuita para sandbox)
- Ngrok (para expor servidor local)

### 2. Configurar o Projeto

```bash
# Clone o projeto
git clone <seu-repositorio>
cd chatbot-pje

# Crie ambiente virtual
python -m venv venv

# Ative o ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale dependências
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente

```bash
# Copie o arquivo de exemplo
cp .env.exemplo .env

# Edite o .env com suas chaves reais
```

**Obter chaves:**
- **Gemini**: https://aistudio.google.com/app/apikey (gratuito)
- **Twilio**: https://console.twilio.com/

---

## Uso

### Teste Local (Terminal)

```bash
python testar.py
```

### Teste Automatizado

```bash
python teste_automatico.py
```

### Servidor WhatsApp

```bash
# Terminal 1: Inicie o servidor
python app.py

# Terminal 2: Exponha com Ngrok
ngrok http 5000
```

Configure o webhook no Twilio:
- URL: `https://seu-id.ngrok.io/whatsapp`
- Método: POST

---

## Estrutura do Projeto

```
├── app.py              # Servidor Flask (WhatsApp + API)
├── rag_engine.py       # Motor RAG com busca híbrida
├── config.py           # Configurações e prompt do sistema
├── conhecimento.txt    # Base de conhecimento (edite aqui!)
├── testar.py           # Chat interativo no terminal
├── teste_automatico.py # Testes automatizados
├── requirements.txt    # Dependências Python
├── .env.exemplo        # Template de configuração
└── .gitignore          # Arquivos ignorados pelo Git
```

---

## Personalização

### Alterar Base de Conhecimento

Edite o arquivo `conhecimento.txt` com as informações do seu sistema. O chatbot só responderá com base neste arquivo.

### Alterar Comportamento do Bot

Edite o `SYSTEM_PROMPT` em `config.py` para ajustar:
- Tom das respostas
- Formatação
- Regras de comportamento

---

## Modelos Disponíveis (Gratuitos)

| Modelo | Limite | Recomendação |
|--------|--------|--------------|
| `gemini-2.5-flash` | 20 req/dia | Melhor qualidade |
| `gemini-flash-latest` | Variável | Alternativa |
| `gemini-2.0-flash-lite` | Maior limite | Alto volume |

Configure em `.env`:
```
GEMINI_MODEL=gemini-2.5-flash
```

---

## Licença

MIT


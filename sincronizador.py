"""
===============================================================================
  SINCRONIZADOR v2.0 — Web Crawler + pgvector + Wiki + Visão + Vídeo
===============================================================================
"""

import os, io, re, time, hashlib, logging, tempfile, urllib.parse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests, urllib3, psycopg2
from bs4 import BeautifulSoup
from PIL import Image
import yt_dlp, whisper
from pgvector.psycopg2 import register_vector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Part

import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 1 — CONFIGURAÇÕES GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"C:\Users\083884131104\Downloads\gen-lang-client-0557303585-fc1e7e8bf20e.json"
)
PROJECT_ID = "gen-lang-client-0557303585"
LOCATION   = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

DB_CONFIG = {
    "dbname": "chatbot_pje", "user": "postgres",
    "password": "1234", "host": "localhost", "port": "5432",
}

URL_INICIAL_PJE  = "https://pjeje.github.io/dicas/"
LIMITE_PAGINAS   = 1000

WIKI_BASE_URL    = "https://guardiao.tre-ma.jus.br"
WIKI_PATH        = "/pje/"
WIKI_USER        = "jadson.santos"
WIKI_PASS        = "mttp0c0sa"
WIKI_PAGINA_RAIZ = "Página_principal"

CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 250
LIMIAR_SIMILARIDADE  = 0.15
EMBEDDING_MODEL_NAME = "text-embedding-004"
VISION_MODEL_NAME    = "gemini-2.5-flash"
WHISPER_MODEL_NAME   = "base"

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 2 — BANCO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

def configurar_banco():
    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documentos_controle (
            url VARCHAR PRIMARY KEY, hash_conteudo VARCHAR(64) NOT NULL,
            atualizado_em TIMESTAMP DEFAULT NOW()
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documentos_vetores (
            id SERIAL PRIMARY KEY,
            url VARCHAR REFERENCES documentos_controle(url) ON DELETE CASCADE,
            conteudo TEXT NOT NULL, embedding VECTOR(768) NOT NULL
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wiki_controle (
            titulo VARCHAR PRIMARY KEY, hash_conteudo VARCHAR(64) NOT NULL,
            atualizado_em TIMESTAMP DEFAULT NOW()
        )""")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wiki_vetores (
            id SERIAL PRIMARY KEY,
            titulo VARCHAR REFERENCES wiki_controle(titulo) ON DELETE CASCADE,
            conteudo TEXT NOT NULL, embedding VECTOR(768) NOT NULL
        )""")

    for t in ("documentos_vetores", "wiki_vetores"):
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{t}_embedding
            ON {t} USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)""")

    # Migração segura para bancos criados pela versão anterior
    cursor.execute("ALTER TABLE documentos_controle ADD COLUMN IF NOT EXISTS atualizado_em TIMESTAMP DEFAULT NOW()")
    cursor.execute("ALTER TABLE wiki_controle ADD COLUMN IF NOT EXISTS atualizado_em TIMESTAMP DEFAULT NOW()")

    conn.commit()
    log.info("[BANCO] Tabelas e índices verificados/criados com sucesso.")
    return conn, cursor

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 3 — UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════════

def gerar_hash(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

def limpar_texto(texto):
    texto = texto.replace("\x00", "")
    texto = re.sub(r"\s{3,}", "\n\n", texto)
    return texto.strip()

def ja_existe_similar(cursor, vetor, tabela="documentos_vetores"):
    cursor.execute(
        f"SELECT embedding <-> %s::vector AS dist FROM {tabela} ORDER BY dist LIMIT 1",
        (vetor,))
    row = cursor.fetchone()
    return bool(row and row[0] < LIMIAR_SIMILARIDADE)

SCOPES_GOOGLE = ['https://www.googleapis.com/auth/drive.readonly']

def autenticar_google():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES_GOOGLE)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES_GOOGLE)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    log.info("[GOOGLE OAUTH] Token de acesso obtido com sucesso.")
    return creds.token # Retornamos a string pura para injetar nos headers

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 4 — EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

def carregar_modelo_embedding():
    log.info(f"[VERTEX AI] Carregando modelo '{EMBEDDING_MODEL_NAME}'...")
    return TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

def gerar_embedding(modelo, texto):
    try:
        return modelo.get_embeddings([texto])[0].values
    except Exception as e:
        log.error(f"[EMBEDDING] Falha: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 5 — IMAGENS (Gemini Vision com contexto)
# ═══════════════════════════════════════════════════════════════════════════

def interpretar_imagem_gemini(url_imagem: str, contexto: str = "") -> str:
    """
    Envia a imagem para o Gemini Vision com o texto que a antecede na página.
    O contexto ajuda a IA a entender do que se trata a imagem (ex: qual tela
    do sistema, qual etapa do processo, etc.) e gera uma descrição mais precisa.

    Args:
        url_imagem : URL da imagem a interpretar.
        contexto   : texto extraído imediatamente antes da imagem na página.
                     Enviado ao Gemini para enriquecer a interpretação.
    """
    try:
        r = requests.get(url_imagem, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=15, verify=False)
        if r.status_code != 200:
            log.debug(f"  [GEMINI VISION] Inacessível: {url_imagem[:60]}")
            return ""

        ext        = urlparse(url_imagem).path.lower().rsplit(".", 1)[-1]
        mimes      = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                      "gif": "image/gif",  "webp": "image/webp", "bmp": "image/bmp"}
        media_type = mimes.get(ext, "image/jpeg")
        parte      = Part.from_data(data=r.content, mime_type=media_type)
        modelo     = GenerativeModel(VISION_MODEL_NAME)

        # Monta o prompt com o contexto da página, se disponível
        if contexto:
            prompt = (
                "Você é um assistente especializado em sistemas jurídicos brasileiros (PJe). "
                "O texto a seguir foi extraído da página do site imediatamente ANTES desta imagem, "
                "e serve de contexto para você entender do que ela trata:\n\n"
                f"--- CONTEXTO DA PÁGINA ---\n{contexto[:800]}\n--- FIM DO CONTEXTO ---\n\n"
                "Com base nesse contexto, descreva detalhadamente o conteúdo da imagem. "
                "Transcreva qualquer texto visível, descreva fluxogramas, formulários, "
                "telas de sistema ou tabelas. Responda em português."
            )
        else:
            prompt = (
                "Você é um assistente especializado em sistemas jurídicos brasileiros (PJe). "
                "Descreva detalhadamente o conteúdo desta imagem. "
                "Transcreva qualquer texto visível, descreva fluxogramas, formulários, "
                "telas de sistema ou tabelas. Responda em português."
            )

        descricao = limpar_texto(modelo.generate_content([parte, prompt]).text)
        log.info(f"  [GEMINI VISION] OK: {url_imagem[:60]}")
        return descricao

    except Exception as e:
        log.warning(f"  [GEMINI VISION] Falha em {url_imagem[:60]}: {e}")
        return ""


def _extrair_contexto_antes_da_imagem(img_tag) -> str:
    """
    Extrai o texto que aparece imediatamente ANTES da tag <img> na página.
    Sobe na árvore HTML para pegar o parágrafo, título ou lista mais próximos.

    Estratégia:
      1. Pega o texto do elemento pai da imagem (ex: <p>, <div>, <figure>).
      2. Se o pai for muito genérico, busca os 3 irmãos anteriores na árvore.
      3. Limita a 500 chars para não poluir o prompt da IA.
    """
    partes = []

    # Texto do elemento pai (ex: legenda dentro de <figure> ou <p>)
    pai = img_tag.parent
    if pai:
        texto_pai = pai.get_text(separator=" ", strip=True)
        # Remove o alt text da própria imagem do contexto
        alt = img_tag.get("alt", "")
        texto_pai = texto_pai.replace(alt, "").strip()
        if texto_pai:
            partes.append(texto_pai)

    # Irmãos anteriores (h1, h2, h3, p, li) — até 3 elementos acima
    elemento = img_tag
    encontrados = 0
    while encontrados < 3:
        elemento = elemento.find_previous_sibling(["h1","h2","h3","h4","p","li","caption","figcaption"])
        if not elemento:
            break
        t = elemento.get_text(strip=True)
        if t:
            partes.insert(0, t)  # mais antigo primeiro
            encontrados += 1

    contexto = " | ".join(p for p in partes if p)
    return contexto[:500]  # limita para não exceder o contexto do prompt


# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 5b — PDFs (Gemini Vision)
# ═══════════════════════════════════════════════════════════════════════════

def interpretar_pdf_gemini(conteudo_pdf_bytes: bytes) -> str:
    """
    Envia o PDF diretamente ao Gemini Vision.
    Funciona tanto com PDFs digitais quanto com PDFs escaneados (OCR nativo).
    """
    try:
        parte  = Part.from_data(data=conteudo_pdf_bytes, mime_type="application/pdf")
        modelo = GenerativeModel(VISION_MODEL_NAME)
        prompt = (
            "Você é um assistente de extração de dados jurídicos. "
            "Leia com atenção este documento PDF. Transcreva todo o texto legível. "
            "Se encontrar tabelas, mantenha a estrutura lógica. Se houver imagens "
            "ou fluxogramas importantes, descreva-os. Responda em português."
        )
        resposta = modelo.generate_content([parte, prompt])
        return limpar_texto(resposta.text)
    except Exception as e:
        log.warning(f"  [GEMINI PDF] Falha na extração: {e}")
        return ""

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 6 — VÍDEOS (yt-dlp + Whisper + Gemini resumo)
# ═══════════════════════════════════════════════════════════════════════════

def _baixar_audio(url_video, destino, google_token=None):
    opcoes_ytdlp = {
        "format": "bestaudio/best", "outtmpl": destino,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True, "no_warnings": True,
        "nocheckcertificate": True,
    }
    
    # Injeta o token para acessar vídeos restritos do Drive da organização
    if google_token:
        opcoes_ytdlp["http_headers"] = {"Authorization": f"Bearer {google_token}"}

    try:
        with yt_dlp.YoutubeDL(opcoes_ytdlp) as ydl:
            ydl.download([url_video])
        return True
    except Exception as e:
        log.warning(f"  [YT-DLP] Falha: {e}")
        return False

def transcrever_audio(caminho_wav):
    try:
        modelo      = whisper.load_model(WHISPER_MODEL_NAME)
        resultado   = modelo.transcribe(caminho_wav, language="pt", fp16=False)
        transcricao = limpar_texto(resultado["text"])
        log.info(f"  [WHISPER] {len(transcricao)} chars transcritos.")
        return transcricao
    except Exception as e:
        log.warning(f"  [WHISPER] Erro: {e}")
        return ""

def resumir_transcricao(transcricao):
    if not transcricao:
        return ""
    try:
        prompt = (
            "Você é um assistente jurídico especializado em sistemas PJe. "
            "Resuma detalhadamente a transcrição abaixo em português, incluindo "
            "pontos principais, passos e termos técnicos:\n\n" + transcricao[:6000]
        )
        return limpar_texto(GenerativeModel(VISION_MODEL_NAME).generate_content(prompt).text)
    except Exception as e:
        log.warning(f"  [RESUMO] Falha: {e}")
        return transcricao

def processar_video(url_video, google_token=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        destino = os.path.join(tmpdir, "audio")
        if not _baixar_audio(url_video, destino, google_token):
            return ""
        wav = destino + ".wav"
        if not os.path.exists(wav):
            candidatos = list(Path(tmpdir).glob("audio*"))
            if not candidatos:
                return ""
            wav = str(candidatos[0])
        return resumir_transcricao(transcrever_audio(wav))

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 7 — CRAWLER PJE DICAS
# ═══════════════════════════════════════════════════════════════════════════

_IMG_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_VID_EXTS     = {".mp4", ".webm", ".ogg", ".avi", ".mov"}
_VID_DOMINIOS = {"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "drive.google.com"}

def _eh_video(url):
    p = urlparse(url)
    return Path(p.path).suffix.lower() in _VID_EXTS or p.netloc in _VID_DOMINIOS

def _eh_imagem(url):
    return Path(urlparse(url).path).suffix.lower() in _IMG_EXTS

def varrer_site(url_inicio, max_paginas):
    log.info(f"\n[ROBÔ] Iniciando varredura: {url_inicio}")
    visitadas, fila, dados = set(), [url_inicio], {}
    dominio_base = urlparse(url_inicio).netloc
    headers      = {"User-Agent": "Mozilla/5.0"}

    while fila and len(visitadas) < max_paginas:
        url_atual = fila.pop(0)
        if url_atual in visitadas:
            continue
        visitadas.add(url_atual)

        try:
            log.info(f"  -> {url_atual}")
            resp = requests.get(url_atual, headers=headers, timeout=15, verify=False)
            if resp.status_code != 200:
                continue

            tipo        = resp.headers.get("Content-Type", "").lower()
            texto_final = ""

            # ── PDF: envia para Gemini ────────────────────────────────────
            if "application/pdf" in tipo or url_atual.lower().endswith(".pdf"):
                log.info(f"     [GEMINI PDF] Transcrevendo: {url_atual[:60]}")
                t = interpretar_pdf_gemini(resp.content)
                if t:
                    texto_final += t

            # ── HTML ──────────────────────────────────────────────────────
            elif "text/html" in tipo:
                soup = BeautifulSoup(resp.content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "aside"]):
                    tag.decompose()

                # Texto principal da página
                texto_final = "\n".join(
                    el.get_text(strip=True)
                    for el in soup.find_all(["h1","h2","h3","h4","p","li","td","th"])
                    if el.get_text(strip=True)
                )

                # Links internos para a fila
                for a in soup.find_all("a", href=True):
                    novo = urljoin(url_atual, a["href"]).split("#")[0]
                    pn   = urlparse(novo)
                    if (pn.netloc == dominio_base and novo not in visitadas
                            and novo not in fila
                            and not pn.path.lower().endswith(
                                (".png",".jpg",".gif",".zip",".rar",".svg"))):
                        fila.append(novo)

                # ── IMAGENS: envia com contexto do texto anterior ─────────
                for img_tag in soup.find_all("img", src=True):
                    ui = urljoin(url_atual, img_tag["src"])
                    if not _eh_imagem(ui):
                        continue

                    # Extrai o texto da página que aparece antes desta imagem
                    contexto = _extrair_contexto_antes_da_imagem(img_tag)
                    if contexto:
                        log.info(f"  [CONTEXTO] '{contexto[:60]}...' → imagem")

                    t = interpretar_imagem_gemini(ui, contexto=contexto)
                    if t:
                        texto_final += f"\n\n[Imagem: {ui}]\n{t}"

                # ── VÍDEOS embed e iframes ────────────────────────────────
                for v in soup.find_all(["video", "source", "iframe"]):
                    src = v.get("src") or v.get("data-src")
                    if src:
                        uv = urljoin(url_atual, src)
                        if _eh_video(uv):
                            t = processar_video(uv)
                            if t:
                                texto_final += f"\n\n[Vídeo: {uv}]\n{t}"

                # ── Links diretos para PDF, imagem ou vídeo ───────────────
                for a in soup.find_all("a", href=True):
                    href = urljoin(url_atual, a["href"]).split("#")[0]
                    if href.lower().endswith(".pdf"):
                        log.info(f"     [GEMINI PDF] Link PDF: {href[:60]}")
                        r2 = requests.get(href, headers=headers, timeout=20, verify=False)
                        if r2.status_code == 200:
                            t = interpretar_pdf_gemini(r2.content)
                            if t:
                                texto_final += f"\n\n[PDF: {href}]\n{t}"
                    elif _eh_imagem(href):
                        contexto = a.get_text(strip=True)[:200]  # texto do link como contexto
                        t = interpretar_imagem_gemini(href, contexto=contexto)
                        if t:
                            texto_final += f"\n\n[Imagem link: {href}]\n{t}"
                    elif _eh_video(href):
                        t = processar_video(href)
                        if t:
                            texto_final += f"\n\n[Vídeo link: {href}]\n{t}"

            if len(texto_final) > 100:
                dados[url_atual] = limpar_texto(texto_final)

        except Exception as e:
            log.error(f"  [ERRO CRAWLER] {url_atual}: {e}")

    log.info(f"\n[ROBÔ] Concluído. {len(dados)} recursos extraídos.")
    return dados

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 8 — MEDIAWIKI (API própria, sem mwclient)
# ═══════════════════════════════════════════════════════════════════════════

def _wiki_api(sess: requests.Session, params: dict) -> dict:
    params["format"] = "json"
    r = sess.post(f"{WIKI_BASE_URL}{WIKI_PATH}api.php", data=params, timeout=30, verify=False)
    r.raise_for_status()
    return r.json()

def conectar_wiki() -> requests.Session:
    log.info(f"[WIKI] Conectando em {WIKI_BASE_URL}{WIKI_PATH}api.php...")
    sess = requests.Session()
    sess.headers.update({"User-Agent": "SincronizadorBot/2.0"})
    sess.verify = False

    # Passo 1: solicita token
    resp1 = _wiki_api(sess, {"action": "login", "lgname": WIKI_USER, "lgpassword": WIKI_PASS})
    res1  = resp1.get("login", {}).get("result")

    if res1 == "NeedToken":
        token = resp1["login"]["token"]
        resp2 = _wiki_api(sess, {"action": "login", "lgname": WIKI_USER,
                                  "lgpassword": WIKI_PASS, "lgtoken": token})
        if resp2.get("login", {}).get("result") != "Success":
            raise Exception(f"Falha de autenticação: {resp2}")
    elif res1 != "Success":
        raise Exception(f"Falha no login (passo 1): {resp1}")

    log.info(f"[WIKI] Autenticado como '{WIKI_USER}'.")
    return sess

def _extrair_texto_html_wiki(html_bruto: str) -> str:
    soup = BeautifulSoup(html_bruto, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return limpar_texto(soup.get_text(separator="\n", strip=True))

def _extrair_pdf_url(url: str, sess: requests.Session, google_token: str = None) -> str:
    """Baixa e transcreve um PDF via Gemini. Suporta autenticação do Google."""
    headers = {}
    if google_token and "google.com" in url:
        headers["Authorization"] = f"Bearer {google_token}"

    try:
        r = sess.get(url, headers=headers, timeout=20, verify=False)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            log.info(f"     [GEMINI PDF] Transcrevendo: {url[:60]}")
            return interpretar_pdf_gemini(r.content)
        log.debug(f"     [PDF] Inacessível ou não é PDF: {url[:60]}")
        return ""
    except Exception as e:
        log.warning(f"     [PDF] Erro em {url[:60]}: {e}")
        return ""

def processar_link_documento(url_ext: str, sess: requests.Session, google_token: str = None) -> str:
    m_drive = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url_ext)
    if m_drive:
        fid = m_drive.group(1)
        log.info(f"     [Google Drive] Baixando ID: {fid}")
        return _extrair_pdf_url(f"https://drive.google.com/uc?export=download&id={fid}", sess, google_token)

    m_docs = re.search(r"docs\.google\.com/document/d/([a-zA-Z0-9_-]+)", url_ext)
    if m_docs:
        did = m_docs.group(1)
        log.info(f"     [Google Docs] Exportando como PDF: {did}")
        return _extrair_pdf_url(f"https://docs.google.com/document/d/{did}/export?format=pdf", sess, google_token)

    if url_ext.lower().split("?")[0].endswith(".pdf"):
        return _extrair_pdf_url(url_ext, sess, google_token)

    return ""

def varrer_wiki(sess: requests.Session, pagina_raiz: str, google_token: str = None) -> dict:
    log.info(f"\n[WIKI] Iniciando varredura a partir de: {pagina_raiz}")
    visitadas, fila, dados = set(), [pagina_raiz], {}

    while fila:
        titulo = fila.pop(0)
        titulo_norm = titulo.replace("_", " ")
        if titulo_norm in visitadas:
            continue
        visitadas.add(titulo_norm)

        try:
            resp = _wiki_api(sess, {
                "action": "parse", "page": titulo,
                "prop": "text|links|images|externallinks", "redirects": "1"
            })
            if "error" in resp:
                continue

            parse_data   = resp.get("parse", {})
            html_bruto   = parse_data.get("text", {}).get("*", "")
            texto_pagina = _extrair_texto_html_wiki(html_bruto) if html_bruto else ""
            log.info(f"  -> [WIKI] {titulo_norm}")

           # ── Links externos: PDFs, Drive, Docs, Vídeos ─────────────────
            for ext_link in parse_data.get("externallinks", []):
                url_ext = ext_link if isinstance(ext_link, str) else ext_link.get("*", "")
                if not url_ext:
                    continue
                # Aqui passamos o token para a leitura de documentos
                t_doc = processar_link_documento(url_ext, sess, google_token)
                if t_doc:
                    texto_pagina += f"\n\n[Documento Externo: {url_ext}]\n{t_doc}"
                    continue
                if _eh_video(url_ext):
                    # Aqui passamos o token para a transcrição de vídeos
                    t_vid = processar_video(url_ext, google_token)
                    if t_vid:
                        texto_pagina += f"\n\n[Vídeo Externo: {url_ext}]\n{t_vid}"

            # ── Arquivos anexados na Wiki (imagens, PDFs, vídeos) ─────────
            soup_wiki = BeautifulSoup(html_bruto, "html.parser") if html_bruto else None

            for nome_arquivo in parse_data.get("images", []):
                # Ignora ícones e SVGs de sistema
                if nome_arquivo.lower().endswith(("svg", "icon.png", "magnify-clip.png")):
                    continue

                url_arquivo = (f"{WIKI_BASE_URL}{WIKI_PATH}index.php"
                               f"?title=Special:FilePath&file={nome_arquivo}")
                ext_arq = Path(nome_arquivo).suffix.lower()

                # ── Imagem: envia com contexto do HTML da página ───────────
                if ext_arq in _IMG_EXTS:
                    # Busca a tag <img> correspondente no HTML para extrair contexto
                    contexto = ""
                    if soup_wiki:
                        tag_img = soup_wiki.find("img", src=re.compile(
                            re.escape(urllib.parse.quote(nome_arquivo.replace(" ", "_"))),
                            re.IGNORECASE
                        ))
                        if tag_img:
                            contexto = _extrair_contexto_antes_da_imagem(tag_img)
                        # Fallback: usa o título da página como contexto mínimo
                        if not contexto:
                            contexto = f"Imagem da página Wiki: {titulo_norm}"

                    log.info(f"     [GEMINI VISION] Imagem Wiki: {nome_arquivo}")
                    t_img = interpretar_imagem_gemini(url_arquivo, contexto=contexto)
                    if t_img:
                        texto_pagina += f"\n\n[Imagem Wiki: {nome_arquivo}]\n{t_img}"

                # ── PDF anexado ────────────────────────────────────────────
                elif ext_arq == ".pdf":
                    log.info(f"     [GEMINI PDF] PDF Wiki: {nome_arquivo}")
                    r = sess.get(url_arquivo, timeout=15, verify=False)
                    if r.status_code == 200:
                        t_pdf = interpretar_pdf_gemini(r.content)
                        if t_pdf:
                            texto_pagina += f"\n\n[PDF Wiki: {nome_arquivo}]\n{t_pdf}"

                # ── Vídeo anexado ──────────────────────────────────────────
                elif ext_arq in _VID_EXTS:
                    log.info(f"     [Vídeo Wiki] Transcrevendo: {nome_arquivo}")
                    try:
                        r = sess.get(url_arquivo, stream=True, timeout=60, verify=False)
                        if r.status_code == 200:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext_arq) as tmp:
                                for chunk in r.iter_content(chunk_size=8192):
                                    tmp.write(chunk)
                                tmp_path = tmp.name
                            resumo = resumir_transcricao(transcrever_audio(tmp_path))
                            os.remove(tmp_path)
                            if resumo:
                                texto_pagina += f"\n\n[Vídeo Wiki: {nome_arquivo}]\n{resumo}"
                    except Exception as e:
                        log.warning(f"  [WIKI VÍDEO] Falha em {nome_arquivo}: {e}")

            if len(texto_pagina) > 50:
                dados[titulo_norm] = texto_pagina
                log.info(f"     [OK] {len(texto_pagina)} chars extraídos.")

            # ── Descoberta de novas páginas (API + HTML bruto) ─────────────
            novos = set()
            for link in parse_data.get("links", []):
                if link.get("ns") == 0 and link.get("*"):
                    novos.add(link["*"])

            if html_bruto:
                soup_links = BeautifulSoup(html_bruto, "html.parser")
                for a_tag in soup_links.find_all("a", href=True):
                    href = a_tag["href"]
                    if f"{WIKI_PATH}index.php/" in href:
                        nome_pag = href.split(f"{WIKI_PATH}index.php/")[-1]
                        nome_pag = nome_pag.split("#")[0].split("?")[0]
                        nome_pag = urllib.parse.unquote(nome_pag).replace("_", " ")
                        ignorar  = ["Special:", "Especial:", "File:", "Arquivo:",
                                    "Talk:", "Discussão:", "User:", "Usuário:"]
                        if nome_pag and not any(nome_pag.startswith(i) for i in ignorar):
                            novos.add(nome_pag)

            for nl in novos:
                if nl not in visitadas and nl not in fila:
                    fila.append(nl)

        except Exception as e:
            log.error(f"  [WIKI ERRO] {titulo_norm}: {e}")

    log.info(f"[WIKI] Concluído. {len(dados)} páginas extraídas.")
    return dados

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 9 — VETORIZAÇÃO E GRAVAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def vetorizar_e_salvar(cursor, conn, modelo_embedding, splitter,
                       chave, texto, tabela_controle, tabela_vetores, campo_chave):
    novo_hash = gerar_hash(texto)
    cursor.execute(
        f"SELECT hash_conteudo FROM {tabela_controle} WHERE {campo_chave} = %s", (chave,))
    row            = cursor.fetchone()
    hash_existente = row[0] if row else None

    if hash_existente == novo_hash:
        log.info(f"  [SEM MUDANÇA] {chave[:70]}")
        return

    if hash_existente:
        log.info(f"  [ATUALIZAR] {chave[:70]}")
        cursor.execute(
            f"UPDATE {tabela_controle} SET hash_conteudo=%s, atualizado_em=NOW() "
            f"WHERE {campo_chave}=%s", (novo_hash, chave))
        cursor.execute(f"DELETE FROM {tabela_vetores} WHERE {campo_chave}=%s", (chave,))
    else:
        log.info(f"  [NOVO] {chave[:70]}")
        cursor.execute(
            f"INSERT INTO {tabela_controle} ({campo_chave}, hash_conteudo) VALUES (%s, %s)",
            (chave, novo_hash))

    inseridos = duplicatas = 0
    for pedaco in splitter.split_text(texto):
        vetor = gerar_embedding(modelo_embedding, pedaco)
        if vetor is None:
            continue
        if ja_existe_similar(cursor, vetor, tabela_vetores):
            duplicatas += 1
            continue
        cursor.execute(
            f"INSERT INTO {tabela_vetores} ({campo_chave}, conteudo, embedding) "
            f"VALUES (%s, %s, %s)", (chave, pedaco, vetor))
        inseridos += 1
        time.sleep(0.05)

    conn.commit()
    log.info(f"     → {inseridos} vetores criados, {duplicatas} duplicatas ignoradas.")

# ═══════════════════════════════════════════════════════════════════════════
#  SEÇÃO 10 — ORQUESTRADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def sincronizar_base():
    log.info("=" * 70)
    log.info("  SINCRONIZADOR v2.0 — Iniciando")
    log.info("=" * 70)

    # Autentica e pega o token ANTES de começar a varredura
    token_google = autenticar_google()

    conn, cursor = configurar_banco()
    modelo_emb   = carregar_modelo_embedding()
    splitter     = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    try:
        # ── Fase 2: PJE Dicas ────────────────────────────────────────────
        log.info("\n[FASE 2] Processando PJE Dicas...")
        for url, texto in varrer_site(URL_INICIAL_PJE, LIMITE_PAGINAS).items():
            vetorizar_e_salvar(cursor, conn, modelo_emb, splitter,
                               chave=url, texto=texto,
                               tabela_controle="documentos_controle",
                               tabela_vetores="documentos_vetores",
                               campo_chave="url")

        # ── Fase 3: Wiki MediaWiki ────────────────────────────────────────
        log.info("\n[FASE 3] Processando Wiki MediaWiki...")
        try:
            sess_wiki = conectar_wiki()
            # Envia o token do Google para a varredura da Wiki
            for titulo, texto in varrer_wiki(sess_wiki, WIKI_PAGINA_RAIZ, token_google).items():
                vetorizar_e_salvar(cursor, conn, modelo_emb, splitter,
                                   chave=titulo, texto=texto,
                                   tabela_controle="wiki_controle",
                                   tabela_vetores="wiki_vetores",
                                   campo_chave="titulo")
        except Exception as e:
            log.error(f"[FASE 3] Falha na Wiki: {e}")

    finally:
        cursor.close()
        conn.close()

    log.info("\n" + "=" * 70)
    log.info("  SINCRONIZAÇÃO CONCLUÍDA COM SUCESSO!")
    log.info("=" * 70)

if __name__ == "__main__":
    sincronizar_base()

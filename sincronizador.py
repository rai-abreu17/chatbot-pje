"""
sincronizador.py - Arquitetura Híbrida: Oracle (controle) + ChromaDB (vetores)
===============================================================================
Oracle: armazena controle de URLs e hashes (dados estruturados)
ChromaDB: armazena embeddings e faz busca vetorial (similaridade semântica)
"""

import os
import oracledb
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configurações
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250

# Oracle - controle de documentos (hashes, URLs)
DB_CONFIG = {
    "user": "CHATBOT_PJE",
    "password": "CHATBOT_PJE_DESE",
    "dsn": oracledb.makedsn("orcldese1", 1521, sid="admteste")
}

# ChromaDB - busca vetorial
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def gerar_hash(texto: str) -> str:
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()

def configurar_banco():
    """Conecta ao Oracle (controle) e ao ChromaDB (vetores)."""
    # Oracle - tabela de controle
    conn = oracledb.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Tabela principal para guardar o controle das URLs e Hashes
    cursor.execute('''
        BEGIN
            EXECUTE IMMEDIATE '
                CREATE TABLE documentos_controle (
                    url VARCHAR2(1000) PRIMARY KEY,
                    hash_conteudo VARCHAR2(64) NOT NULL
                )
            ';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -955 THEN NULL;
                ELSE RAISE;
                END IF;
        END;
    ''')
    
    conn.commit()

    # ChromaDB - coleção de vetores
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name="documentos_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    return conn, cursor, collection

# Lista de URLs oficiais fornecida pelo Arquiteto (Você!)
URLS_OFICIAIS = [
    "https://pjeje.github.io/dicas/acesso/",
    "https://pjeje.github.io/dicas/advogados/",
    "https://pjeje.github.io/dicas/atos/",
    "https://pjeje.github.io/dicas/autos/",
    "https://pjeje.github.io/dicas/autuacao/",
    "https://pjeje.github.io/dicas/classes/",
    "https://pjeje.github.io/dicas/comunicacao/",
    "https://pjeje.github.io/dicas/consulta/",
    "https://pjeje.github.io/dicas/defensorias/",
    "https://pjeje.github.io/dicas/distribuicao/",
    "https://pjeje.github.io/dicas/etiquetas/",
    "https://pjeje.github.io/dicas/papeis/",
    "https://pjeje.github.io/dicas/prazos/",
    "https://pjeje.github.io/dicas/procuradorias/",
    "https://pjeje.github.io/dicas/recursos/",
    "https://pjeje.github.io/dicas/remessa/",
    "https://pjeje.github.io/dicas/sessaojulg/",
    "https://pjeje.github.io/dicas/sigilo/",
    "https://pjeje.github.io/dicas/manual/",
    "https://pjeje.github.io/dicas/automacao/",
    "https://pjeje.github.io/dicas/tramitacao/"
]

def extrair_dados_do_site():
    print("[EXTRAÇÃO] Lendo dados locais e conectando com URLs Oficiais...")
    dados_secoes = {}
    try:
        with open("conhecimento.txt", "r", encoding="utf-8") as f:
            texto_completo = f.read()
            
        # Divide o texto usando "SEÇÃO " como tesoura
        secoes = texto_completo.split("SEÇÃO ")
        
        # O for agora usa 'enumerate' para saber qual é o índice (0, 1, 2...)
        for indice, secao in enumerate(secoes[1:]):
            numero_secao = secao.split(":")[0].strip()
            
            # Pega a URL oficial correspondente a este índice
            # Se por acaso tiver mais seções que URLs, ele cria uma temporária para não quebrar
            if indice < len(URLS_OFICIAIS):
                url_secao = URLS_OFICIAIS[indice]
            else:
                url_secao = f"https://pjeje.github.io/dicas/secao_extra_{numero_secao}"
            
            # Reconstrói o texto
            texto_secao = "SEÇÃO " + secao.strip()
            
            # Guarda no banco de dados temporário (Dicionário)
            dados_secoes[url_secao] = texto_secao
            print(f"  -> Seção {numero_secao} salva em: {url_secao}")
            
        return dados_secoes
        
    except FileNotFoundError:
        print("Erro: arquivo conhecimento.txt não encontrado.")
        return {}

def sincronizar_base():
    print("Iniciando a Arquitetura Híbrida (Oracle + ChromaDB)...")
    
    conn, cursor, collection = configurar_banco()
    
    print("[IA] Carregando o modelo de transformação de texto em vetores...")
    # Baixa um modelo leve e rápido, ideal para português
    modelo = SentenceTransformer('all-MiniLM-L6-v2') 
    
    dados_site = extrair_dados_do_site()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    # Consulta o Oracle para saber o que já existe
    cursor.execute('SELECT url, hash_conteudo FROM documentos_controle')
    documentos_locais = {row[0]: row[1] for row in cursor.fetchall()}

    chunk_counter = 0  # Contador para gerar IDs únicos no ChromaDB

    for url, texto_novo in dados_site.items():
        novo_hash = gerar_hash(texto_novo)
        
        # Se for um documento novo ou atualizado, processamos!
        if url not in documentos_locais or documentos_locais[url] != novo_hash:
            if url in documentos_locais:
                print(f"[ATUALIZAR] Recriando vetores para: {url}")
                # Remove chunks antigos do ChromaDB filtrando pela URL
                ids_antigos = collection.get(where={"url": url})["ids"]
                if ids_antigos:
                    collection.delete(ids=ids_antigos)
                # Atualiza hash no Oracle
                cursor.execute('UPDATE documentos_controle SET hash_conteudo = :1 WHERE url = :2', (novo_hash, url))
            else:
                print(f"[NOVO] Criando vetores para: {url}")
                # Insere controle no Oracle
                cursor.execute('INSERT INTO documentos_controle (url, hash_conteudo) VALUES (:1, :2)', (url, novo_hash))
            
            # Divide o texto e cria os vetores no ChromaDB
            pedacos = splitter.split_text(texto_novo)
            for pedaco in pedacos:
                vetor = modelo.encode(pedaco).tolist()  # Mágica: Texto vira números!
                chunk_id = f"{url}__chunk_{chunk_counter}"
                chunk_counter += 1
                
                collection.add(
                    ids=[chunk_id],
                    embeddings=[vetor],
                    documents=[pedaco],
                    metadatas=[{"url": url}]
                )

    conn.commit()
    cursor.close()
    conn.close()
    print("Sincronização concluída com Sucesso Absoluto!")

if __name__ == "__main__":
    sincronizar_base()
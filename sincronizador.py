"""
sincronizador.py - Arquitetura Definitiva com PostgreSQL + pgvector
===================================================================
"""

import psycopg2
import hashlib
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configurações
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250

DB_CONFIG = {
    "dbname": "base_conhecimento_pje",
    "user": "postgres",                
    "password": "minhasenha123",       # A senha que colocamos no Docker
    "host": "localhost",               
    "port": "5433"                     # A porta nova que definimos no Docker!
}

def gerar_hash(texto: str) -> str:
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()

def configurar_banco():
    """Conecta, liga a extensão de vetores e cria as tabelas."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Ativa o superpoder dos vetores no banco
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Tabela principal para guardar o controle das URLs e Hashes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos_controle (
            url TEXT PRIMARY KEY,
            hash_conteudo TEXT NOT NULL
        )
    ''')
    
    # Tabela para guardar os pedaços de texto e seus vetores
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos_chunks (
            id SERIAL PRIMARY KEY,
            url TEXT,
            texto_chunk TEXT NOT NULL,
            embedding vector(384) 
        )
    ''')
    
    conn.commit()
    register_vector(conn) # Ensina o Python a ler/escrever vetores
    return conn, cursor

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
    print("Iniciando a Arquitetura Suprema (PostgreSQL + pgvector)...")
    
    conn, cursor = configurar_banco()
    
    print("[IA] Carregando o modelo de transformação de texto em vetores...")
    # Baixa um modelo leve e rápido, ideal para português
    modelo = SentenceTransformer('all-MiniLM-L6-v2') 
    
    dados_site = extrair_dados_do_site()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    cursor.execute('SELECT url, hash_conteudo FROM documentos_controle')
    documentos_locais = {row[0]: row[1] for row in cursor.fetchall()}

    for url, texto_novo in dados_site.items():
        novo_hash = gerar_hash(texto_novo)
        
        # Se for um documento novo ou atualizado, processamos!
        if url not in documentos_locais or documentos_locais[url] != novo_hash:
            if url in documentos_locais:
                print(f"[ATUALIZAR] Recriando vetores para: {url}")
                cursor.execute('DELETE FROM documentos_chunks WHERE url = %s', (url,))
                cursor.execute('UPDATE documentos_controle SET hash_conteudo = %s WHERE url = %s', (novo_hash, url))
            else:
                print(f"[NOVO] Criando vetores para: {url}")
                cursor.execute('INSERT INTO documentos_controle (url, hash_conteudo) VALUES (%s, %s)', (url, novo_hash))
            
            # Divide o texto e cria os vetores
            pedacos = splitter.split_text(texto_novo)
            for pedaco in pedacos:
                vetor = modelo.encode(pedaco).tolist() # Mágica: Texto vira números!
                cursor.execute(
                    'INSERT INTO documentos_chunks (url, texto_chunk, embedding) VALUES (%s, %s, %s)',
                    (url, pedaco, vetor)
                )

    conn.commit()
    cursor.close()
    conn.close()
    print("Sincronização concluída com Sucesso Absoluto!")

if __name__ == "__main__":
    sincronizar_base()
"""
rag_engine.py - Motor de Recuperação Aumentada por Geração (RAG) APRIMORADO
============================================================================
Este módulo é responsável por:
1. Ler o arquivo de conhecimento (conhecimento.txt)
2. Dividir o texto em pedaços menores (chunks) preservando seções
3. Criar embeddings usando busca híbrida
4. Buscar os trechos mais relevantes usando busca híbrida
5. Expandir consultas com sinônimos para melhor match
6. Enviar a pergunta + contexto para o Google Gemini e retornar a resposta

Melhorias implementadas:
- Expansão de consulta com sinônimos
- Busca híbrida (semântica + keywords)
- Metadados nos chunks para melhor contexto
- Filtragem por relevância mínima
"""

import os
import re
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    KNOWLEDGE_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    SYSTEM_PROMPT,
)


# ============================================================
# DICIONÁRIO DE SINÔNIMOS PARA EXPANSÃO DE CONSULTA
# ============================================================
SINONIMOS = {
    # Problemas de desempenho
    "lento": ["lentidão", "devagar", "demora", "travando", "trava", "lerdo", "parado", "demorado"],
    "lentidão": ["lento", "devagar", "demora", "travando", "lerdo", "demorado"],
    "travando": ["lento", "trava", "congela", "paralisa", "não responde", "travou"],
    "demorado": ["lento", "lentidão", "devagar", "demora"],
    
    # Problemas de acesso
    "login": ["entrar", "acessar", "logar", "autenticar", "acesso"],
    "entrar": ["login", "acessar", "logar", "autenticar"],
    "não entra": ["não consigo entrar", "não loga", "não acessa", "erro login"],
    "senha": ["password", "credencial", "chave", "código"],
    "acesso": ["login", "entrar", "acessar", "logar", "autenticar"],
    
    # Certificado digital
    "certificado": ["token", "smart card", "assinatura digital", "e-cpf", "e-cnpj"],
    "token": ["certificado", "certificado digital", "pendrive"],
    
    # Interface
    "branco": ["vazio", "não carrega", "página em branco", "tela branca"],
    "erro": ["problema", "falha", "bug", "não funciona"],
    "pisca": ["atualiza", "recarrega", "volta", "piscando", "intermitente"],
    
    # Documentos
    "anexar": ["juntar", "subir", "upload", "enviar arquivo"],
    "documento": ["arquivo", "pdf", "petição", "anexo"],
    "petição": ["peticionar", "protocolar", "documento"],
    
    # Sistema
    "sistema": ["pje", "processo judicial eletrônico"],
    "cache": ["histórico", "dados", "memória", "cookies"],
    "limpar": ["apagar", "excluir", "remover", "deletar"],
    
    # Navegador
    "navegador": ["browser", "firefox", "chrome", "internet explorer", "edge"],
    "firefox": ["navegador", "mozilla", "browser"],
    "chrome": ["navegador", "google chrome", "browser"],
    
    # Consulta
    "consultar": ["pesquisar", "buscar", "procurar", "ver", "achar"],
    "processo": ["autos", "ação", "procedimento"],
    
    # Primeiro acesso
    "primeiro": ["inicial", "1º", "primeira vez"],
}


class RAGEngine:
    """
    Motor RAG aprimorado que gerencia a base de conhecimento e gera respostas.
    
    Uso básico:
        engine = RAGEngine()
        resposta = engine.responder("Como limpar o cache?")
    """

    def __init__(self):
        """Inicializa o motor RAG: carrega documentos, cria índice vetorial."""
        print("[RAG] Inicializando motor RAG aprimorado...")

        # Configura o cliente Google Gemini (nova API)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.modelo_nome = GEMINI_MODEL

        # Banco vetorial ChromaDB (em memória para simplicidade)
        self.chroma_client = chromadb.Client()

        # ============================================================
        # CONFIGURAÇÃO DE EMBEDDINGS - Usando embedding local do ChromaDB
        # ============================================================
        # ChromaDB usa all-MiniLM-L6-v2 por padrão (bom para português)
        # ============================================================

        # Cria ou obtém a coleção (sem embedding function personalizada)
        self.collection = self.chroma_client.get_or_create_collection(
            name="base_conhecimento",
        )

        # Carrega e indexa os documentos
        self._carregar_documentos()
        print(f"[RAG] Motor inicializado com {self.collection.count()} trechos indexados.")

    def _extrair_secao(self, texto: str) -> str:
        """Extrai o título da seção de um trecho de texto."""
        # Procura por padrões de seção como "SEÇÃO X:" ou "X.X -"
        match = re.search(r'(SEÇÃO \d+[^:]*:|^\d+\.\d+\s*-[^\n]+)', texto, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Geral"

    def _carregar_documentos(self):
        """
        Lê o arquivo de conhecimento, divide em pedaços e indexa no ChromaDB.
        Preserva metadados como seção para melhor contexto.
        """
        # Verifica se o arquivo existe
        if not os.path.exists(KNOWLEDGE_FILE):
            raise FileNotFoundError(
                f"Arquivo de conhecimento não encontrado: {KNOWLEDGE_FILE}\n"
                f"Crie o arquivo 'conhecimento.txt' na raiz do projeto."
            )

        # Lê o conteúdo do arquivo
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            texto_completo = f.read()

        print(f"[RAG] Arquivo carregado: {len(texto_completo)} caracteres.")

        # Divide o texto em pedaços menores (chunks)
        # Separadores priorizados para manter seções inteiras
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n---\n",      # Separador de seção principal
                "\nSEÇÃO",      # Início de seção
                "\n\n",         # Parágrafos
                "\n",           # Linhas
                ". ",           # Frases
                " ",            # Palavras
            ],
            keep_separator=True,
        )
        pedacos = splitter.split_text(texto_completo)

        print(f"[RAG] Texto dividido em {len(pedacos)} pedaços.")

        # Limpa a coleção anterior (caso esteja reiniciando)
        if self.collection.count() > 0:
            ids_existentes = self.collection.get()["ids"]
            self.collection.delete(ids=ids_existentes)

        # Prepara metadados para cada chunk
        metadados = []
        for pedaco in pedacos:
            secao = self._extrair_secao(pedaco)
            # Extrai palavras-chave do chunk para busca híbrida
            palavras = set(re.findall(r'\b[a-záéíóúâêîôûãõç]{4,}\b', pedaco.lower()))
            metadados.append({
                "secao": secao,
                "keywords": " ".join(list(palavras)[:20]),  # Top 20 palavras
                "tamanho": len(pedaco),
            })

        # Adiciona os pedaços ao ChromaDB com metadados
        ids = [f"chunk_{i}" for i in range(len(pedacos))]
        self.collection.add(
            documents=pedacos,
            ids=ids,
            metadatas=metadados,
        )

    def _expandir_consulta(self, pergunta: str) -> str:
        """
        Expande a consulta com sinônimos para melhorar o match.
        
        Args:
            pergunta: A pergunta original do usuário.
            
        Returns:
            Pergunta expandida com termos relacionados.
        """
        # Remove pontuação para melhor matching de sinônimos
        texto_limpo = re.sub(r'[^\w\s]', '', pergunta.lower())
        palavras = texto_limpo.split()
        termos_expandidos = set(palavras)
        
        for palavra in palavras:
            # Adiciona sinônimos se existirem
            if palavra in SINONIMOS:
                termos_expandidos.update(SINONIMOS[palavra])
            
            # Busca reversa: se a palavra é sinônimo de algo
            for chave, sinonimos in SINONIMOS.items():
                if palavra in sinonimos:
                    termos_expandidos.add(chave)
                    termos_expandidos.update(sinonimos)
        
        consulta_expandida = pergunta + " " + " ".join(termos_expandidos)
        return consulta_expandida

    def _calcular_score_keywords(self, pergunta: str, documento: str) -> float:
        """
        Calcula um score baseado em keywords (busca lexical).
        Complementa a busca semântica.
        Também considera sinônimos das palavras da pergunta.
        """
        # Remove pontuação
        pergunta_limpa = re.sub(r'[^\w\s]', '', pergunta.lower())
        palavras_pergunta_original = set(re.findall(r'\b[a-záéíóúâêîôûãõç]{3,}\b', pergunta_limpa))
        palavras_doc = set(re.findall(r'\b[a-záéíóúâêîôûãõç]{3,}\b', documento.lower()))
        
        if not palavras_pergunta_original:
            return 0.0
        
        # Expande as palavras da pergunta com sinônimos
        palavras_expandidas = set(palavras_pergunta_original)
        for palavra in palavras_pergunta_original:
            if palavra in SINONIMOS:
                palavras_expandidas.update(SINONIMOS[palavra])
            # Busca reversa
            for chave, sinonimos in SINONIMOS.items():
                if palavra in sinonimos:
                    palavras_expandidas.add(chave)
                    palavras_expandidas.update(sinonimos)
        
        # Intersecao considerando sinônimos
        intersecao = palavras_expandidas & palavras_doc
        uniao = palavras_expandidas | palavras_doc
        
        jaccard = len(intersecao) / len(uniao) if uniao else 0
        
        # Bonus para termos-chave encontrados no documento
        termos_importantes = {"cache", "senha", "certificado", "login", "erro", "lento", "lentidão", 
                              "navegador", "firefox", "chrome", "primeiro", "acesso", "token", "pje",
                              "problema", "solução", "limpar", "travando", "devagar"}
        bonus = sum(0.15 for t in termos_importantes if t in intersecao)
        
        return min(jaccard + bonus, 1.0)

    def buscar_contexto(self, pergunta: str) -> str:
        """
        Busca os trechos mais relevantes usando busca híbrida.
        Combina busca semântica (embeddings) com busca por keywords.
        
        Args:
            pergunta: A pergunta do usuário.
            
        Returns:
            String com os trechos relevantes concatenados.
        """
        # Expande a consulta com sinônimos
        consulta_expandida = self._expandir_consulta(pergunta)
        print(f"[RAG] Consulta expandida: {consulta_expandida[:100]}...")
        
        # Busca semântica no ChromaDB (busca TODOS os documentos para reranking híbrido)
        total_docs = self.collection.count()
        resultados = self.collection.query(
            query_texts=[consulta_expandida],
            n_results=total_docs,  # Busca todos para reranking híbrido
            include=["documents", "distances", "metadatas"],
        )

        if not resultados["documents"] or not resultados["documents"][0]:
            return ""

        # Combina score semântico com score de keywords (busca híbrida)
        documentos = resultados["documents"][0]
        distancias = resultados["distances"][0] if resultados["distances"] else [1.0] * len(documentos)
        
        scored_docs = []
        for i, (doc, dist) in enumerate(zip(documentos, distancias)):
            # Score semântico (converte distância em similaridade)
            score_semantico = max(0, 1 - dist)
            
            # Score de keywords
            score_keywords = self._calcular_score_keywords(pergunta, doc)
            
            # Score híbrido (50% semântico, 50% keywords - balanceado para melhor match em português)
            score_final = 0.5 * score_semantico + 0.5 * score_keywords
            
            scored_docs.append((doc, score_final))
        
        # Ordena por score final e pega os top K
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:TOP_K_RESULTS]
        
        # Filtra documentos com score muito baixo
        LIMIAR_RELEVANCIA = 0.08  # Limiar mais baixo para capturar mais resultados
        docs_relevantes = [doc for doc, score in top_docs if score >= LIMIAR_RELEVANCIA]
        
        if not docs_relevantes:
            # Se nenhum passou no limiar, pega pelo menos o melhor
            docs_relevantes = [top_docs[0][0]] if top_docs else []
        
        print(f"[RAG] {len(docs_relevantes)} trechos relevantes encontrados")
        
        contexto = "\n\n---\n\n".join(docs_relevantes)
        return contexto

    def responder(self, pergunta: str) -> str:
        """
        Gera uma resposta para a pergunta do usuário usando RAG aprimorado.
        
        Fluxo:
        1. Expande a consulta com sinônimos
        2. Busca trechos relevantes (híbrido: semântico + keywords)
        3. Monta o prompt com o contexto
        4. Envia para o Google Gemini
        5. Retorna a resposta
        
        Args:
            pergunta: A pergunta do usuário.
            
        Returns:
            String com a resposta gerada pela IA.
        """
        # Passo 1-2: Buscar contexto relevante (já inclui expansão)
        contexto = self.buscar_contexto(pergunta)

        if not contexto:
            return (
                "Desculpe, não encontrei informações relevantes na minha "
                "base de conhecimento. Por favor, entre em contato com o "
                "suporte técnico do seu tribunal."
            )

        # Passo 3: Montar o prompt do sistema com o contexto
        prompt_sistema = SYSTEM_PROMPT.format(contexto=contexto)

        # Passo 4: Enviar para o Google Gemini (nova API) com retry
        import time
        max_tentativas = 3
        
        for tentativa in range(max_tentativas):
            try:
                prompt_completo = f"{prompt_sistema}\n\nPergunta do usuário: {pergunta}"
                resposta = self.client.models.generate_content(
                    model=self.modelo_nome,
                    contents=prompt_completo,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=1024,  # Aumentado para garantir respostas completas
                    ),
                )
                return resposta.text

            except Exception as e:
                erro_str = str(e)
                print(f"[RAG] Erro ao chamar Gemini (tentativa {tentativa + 1}/{max_tentativas}): {e}")
                
                # Retry automático para erros 503 (high demand) e 429 (rate limit)
                if ("503" in erro_str or "429" in erro_str or "UNAVAILABLE" in erro_str) and tentativa < max_tentativas - 1:
                    tempo_espera = (tentativa + 1) * 2  # 2s, 4s, 6s
                    print(f"[RAG] Aguardando {tempo_espera}s antes de tentar novamente...")
                    time.sleep(tempo_espera)
                    continue
                
                return (
                    "Desculpe, ocorreu um erro ao processar sua pergunta. "
                    "Por favor, tente novamente em alguns instantes."
                )
        
        return "Desculpe, o sistema está sobrecarregado. Tente novamente em alguns segundos."

    def recarregar_documentos(self):
        """
        Recarrega a base de conhecimento sem reiniciar o servidor.
        Útil quando o arquivo conhecimento.txt é atualizado.
        """
        print("[RAG] Recarregando base de conhecimento...")
        self._carregar_documentos()
        print(f"[RAG] Base recarregada: {self.collection.count()} trechos.")

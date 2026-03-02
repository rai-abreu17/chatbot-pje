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
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RESULTS,
    SYSTEM_PROMPT,
)

# Diretório onde o ChromaDB persiste os dados vetoriais
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


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
    "senha": ["password", "credencial", "chave", "código", "esqueci", "bloqueada", "recuperar"],
    "esqueci": ["esquecida", "perdi", "não lembro", "esqueceu", "senha"],
    "bloqueada": ["bloqueou", "bloqueado", "travada", "travou"],
    "acesso": ["login", "entrar", "acessar", "logar", "autenticar"],
    "usuário": ["user", "usuario", "conta", "cadastro"],
    "encontrado": ["encontrar", "reconhecido", "localizado", "achado"],
    "não encontrado": ["usuário não encontrado", "erro usuário", "conta não existe"],
    
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
        """Inicializa o motor RAG: conecta ao ChromaDB e prepara o modelo de vetores."""
        print("[RAG] A inicializar motor RAG Supremo com ChromaDB...")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.modelo_nome = GEMINI_MODEL

        # Conecta ao ChromaDB (busca vetorial)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name="documentos_chunks",
            metadata={"hnsw:space": "cosine"}
        )

        # Carrega a IA que transforma perguntas em números
        print("[RAG] A carregar modelo de vetores...")
        self.modelo_vetores = SentenceTransformer('all-MiniLM-L6-v2')
        print("[RAG] Motor pronto e conectado ao ChromaDB!")



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

    def buscar_contexto(self, pergunta: str) -> str:
        """Busca os trechos mais relevantes usando ChromaDB (busca vetorial nativa)."""
        
        # Expande a consulta com sinónimos
        consulta_expandida = self._expandir_consulta(pergunta) if hasattr(self, '_expandir_consulta') else pergunta
        print(f"[RAG] A procurar por: {consulta_expandida[:100]}...")

        # Transforma a pergunta do utilizador num vetor (lista de números)
        vetor_pergunta = self.modelo_vetores.encode(consulta_expandida).tolist()

        # Busca vetorial no ChromaDB (similaridade de cosseno nativa)
        resultados = self.collection.query(
            query_embeddings=[vetor_pergunta],
            n_results=TOP_K_RESULTS,
            include=["documents", "metadatas"]
        )

        if not resultados or not resultados["documents"] or not resultados["documents"][0]:
            return ""

        docs_relevantes = []
        for texto, metadata in zip(resultados["documents"][0], resultados["metadatas"][0]):
            url = metadata.get("url", "desconhecida")
            # Injetamos a URL da fonte junto com o texto para o Gemini saber de onde veio!
            docs_relevantes.append(f"[Fonte: {url}]\n{texto}")
            print(f"[RAG] Encontrado trecho em: {url}")

        contexto = "\n\n---\n\n".join(docs_relevantes)
        return contexto

    def responder(self, pergunta: str, historico_conversa: str = "") -> str:
        """
        Gera uma resposta para a pergunta do usuário usando RAG aprimorado.
        
        Fluxo:
        1. Expande a consulta com sinônimos (e histórico se houver)
        2. Busca trechos relevantes (híbrido: semântico + keywords)
        3. Monta o prompt com o contexto
        4. Envia para o Google Gemini
        5. Retorna a resposta
        
        Args:
            pergunta: A pergunta do usuário.
            historico_conversa: Histórico da conversa para contexto (opcional).
            
        Returns:
            String com a resposta gerada pela IA.
        """
        # Se houver histórico, enriquece a busca com o contexto anterior
        consulta_para_busca = pergunta
        if historico_conversa:
            # Extrai palavras-chave do histórico para melhorar a busca
            palavras_historico = re.findall(r'\b[a-záéíóúâêîôûãõç]{4,}\b', historico_conversa.lower())
            termos_relevantes = {"cache", "limpar", "chrome", "firefox", "edge", "navegador", "senha", 
                                 "certificado", "login", "acesso", "lentidão", "lento", "erro",
                                 "token", "pje", "primeiro", "acesso", "esqueci", "esquecida",
                                 "bloqueada", "recuperar", "solicitar", "nova", "google", "usuário",
                                 "encontrado", "link", "expirado"}
            palavras_contexto = [p for p in palavras_historico if p in termos_relevantes]
            if palavras_contexto:
                consulta_para_busca = f"{pergunta} {' '.join(set(palavras_contexto))}"
                print(f"[RAG] Consulta enriquecida com contexto: {consulta_para_busca}")
        
        # Passo 1-2: Buscar contexto relevante (já inclui expansão)
        contexto = self.buscar_contexto(consulta_para_busca)

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
                # Inclui histórico da conversa se houver
                if historico_conversa:
                    prompt_completo = f"{prompt_sistema}\n\nCONTEXTO DA CONVERSA (use estas informações para decidir se precisa perguntar algo):\n{historico_conversa}\n\nMensagem atual do usuário: {pergunta}"
                else:
                    prompt_completo = f"{prompt_sistema}\n\nMensagem do usuário: {pergunta}"
                
                resposta = self.client.models.generate_content(
                    model=self.modelo_nome,
                    contents=prompt_completo,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=4096,  # Aumentado para evitar respostas cortadas
                    ),
                )
                
                # Verifica se a resposta foi truncada
                texto_resposta = resposta.text or ""
                
                # Log do finish_reason para debug
                if resposta.candidates:
                    finish_reason = resposta.candidates[0].finish_reason
                    print(f"[RAG] Finish reason: {finish_reason}")
                    if str(finish_reason) == "MAX_TOKENS":
                        print(f"[RAG] AVISO: Resposta truncada por limite de tokens!")
                
                # Remove markdown que o Gemini insiste em usar
                texto_resposta = texto_resposta.replace('**', '').replace('*', '').replace('__', '').replace('`', '')
                
                return texto_resposta

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
        Recarrega a base de conhecimento executando o sincronizador.
        Útil quando o arquivo conhecimento.txt é atualizado.
        """
        print("[RAG] Para recarregar, execute: python sincronizador.py")

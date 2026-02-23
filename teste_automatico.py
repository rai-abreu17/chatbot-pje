"""
teste_automatico.py - Testes Automatizados do Chatbot PJe
============================================================
Este script executa uma bateria de perguntas predefinidas
para validar que o chatbot está funcionando corretamente.

Uso:
    python teste_automatico.py

Ele testa:
1. Perguntas que TÊM resposta na base de conhecimento
2. Perguntas que NÃO TÊM resposta (deve recusar educadamente)
3. Tempo de resposta de cada pergunta
"""

import time
from rag_engine import RAGEngine


# Perguntas de teste
PERGUNTAS_TESTE = [
    # Perguntas que DEVEM ser respondidas com base no documento
    {
        "pergunta": "Como limpar o cache do Google Chrome?",
        "deve_conter": ["configurações", "privacidade", "cache", "limpar", "remover"],
        "tipo": "resposta_esperada",
    },
    {
        "pergunta": "O que fazer quando o sistema está lento?",
        "deve_conter": ["cache", "limpar", "navegador", "lentidão"],
        "tipo": "resposta_esperada",
    },
    {
        "pergunta": "Esqueci minha senha, como recuperar?",
        "deve_conter": ["senha", "solicitar", "e-mail", "link"],
        "tipo": "resposta_esperada",
    },
    {
        "pergunta": "Qual navegador devo usar para acessar o PJe?",
        "deve_conter": ["Firefox", "Chrome", "navegador"],
        "tipo": "resposta_esperada",
    },
    {
        "pergunta": "Preciso de certificado digital para usar o PJe?",
        "deve_conter": ["certificado", "digital", "token", "PJe Office"],
        "tipo": "resposta_esperada",
    },
    {
        "pergunta": "A tela pisca e volta quando tento logar com certificado",
        "deve_conter": ["PJe Office", "autorizado", "senha", "sites"],
        "tipo": "resposta_esperada",
    },
    {
        "pergunta": "Como faço o primeiro acesso ao sistema?",
        "deve_conter": ["primeiro", "acesso", "popup", "cadastro", "certificado"],
        "tipo": "resposta_esperada",
    },
    # Perguntas que NÃO devem ser respondidas (fora do escopo)
    {
        "pergunta": "Qual a receita de bolo de chocolate?",
        "deve_conter": ["não encontrei", "suporte", "base de conhecimento", "minha função", "não está relacionado", "não posso ajudar", "pje"],
        "tipo": "recusa_esperada",
    },
    {
        "pergunta": "Quem é o presidente do Brasil?",
        "deve_conter": ["não encontrei", "suporte", "base de conhecimento", "minha função", "não está relacionado", "não posso ajudar", "pje"],
        "tipo": "recusa_esperada",
    },
]


def executar_testes():
    """Executa todos os testes e exibe os resultados."""
    print("=" * 60)
    print("  TESTES AUTOMATIZADOS - Chatbot PJe")
    print("=" * 60)
    print()

    # Inicializa o motor
    engine = RAGEngine()
    print()

    total = len(PERGUNTAS_TESTE)
    aprovados = 0
    reprovados = 0
    
    # Delay entre requisições para evitar rate limit do Gemini (5 req/min no plano gratuito)
    DELAY_ENTRE_TESTES = 13  # 13 segundos = ~4.6 req/min, dentro do limite

    for i, teste in enumerate(PERGUNTAS_TESTE, 1):
        pergunta = teste["pergunta"]
        palavras_esperadas = teste["deve_conter"]
        tipo = teste["tipo"]

        print(f"Teste {i}/{total}: {pergunta}")

        # Mede o tempo de resposta
        inicio = time.time()
        resposta = engine.responder(pergunta)
        tempo = time.time() - inicio

        # Verifica se a resposta contém pelo menos uma palavra esperada
        resposta_lower = resposta.lower()
        encontradas = [p for p in palavras_esperadas if p.lower() in resposta_lower]

        if encontradas:
            status = "✓ APROVADO"
            aprovados += 1
        else:
            status = "✗ REPROVADO"
            reprovados += 1

        print(f"  Tipo: {tipo}")
        print(f"  Resposta: {resposta[:150]}...")
        print(f"  Palavras encontradas: {encontradas}")
        print(f"  Tempo: {tempo:.2f}s")
        print(f"  Status: {status}")
        print()
        
        # Aguarda antes da próxima requisição para evitar rate limit
        if i < total:
            print(f"  [Aguardando {DELAY_ENTRE_TESTES}s para evitar rate limit...]")
            time.sleep(DELAY_ENTRE_TESTES)
            print()

    # Resumo
    print("=" * 60)
    print(f"  RESULTADO: {aprovados}/{total} testes aprovados")
    print(f"  Aprovados: {aprovados} | Reprovados: {reprovados}")
    taxa = (aprovados / total) * 100 if total > 0 else 0
    print(f"  Taxa de sucesso: {taxa:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    executar_testes()

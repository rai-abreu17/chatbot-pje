"""
testar.py - Script de Teste Interativo do Chatbot PJe
========================================================
Execute este script para conversar com o chatbot diretamente
no terminal, sem precisar do WhatsApp ou do servidor Flask.

Uso:
    python testar.py

O script irá:
1. Carregar a base de conhecimento
2. Iniciar um chat interativo no terminal
3. Você digita perguntas e recebe respostas

Digite 'sair' para encerrar.
"""

from rag_engine import RAGEngine


def main():
    print("=" * 60)
    print("  CHATBOT ASSISTENTE PJe - Modo de Teste")
    print("=" * 60)
    print()

    # Inicializa o motor RAG
    engine = RAGEngine()

    print()
    print("-" * 60)
    print("Chat iniciado! Digite sua pergunta e pressione Enter.")
    print("Digite 'sair' para encerrar.")
    print("Digite 'recarregar' para atualizar a base de conhecimento.")
    print("-" * 60)
    print()

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nChat encerrado. Até logo!")
            break

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("\nChat encerrado. Até logo!")
            break

        if pergunta.lower() == "recarregar":
            engine.recarregar_documentos()
            print("Base de conhecimento recarregada!\n")
            continue

        # Gera a resposta
        resposta = engine.responder(pergunta)
        print(f"\nAssistente SeSJU: {resposta}\n")


if __name__ == "__main__":
    main()

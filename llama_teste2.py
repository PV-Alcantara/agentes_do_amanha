from langchain_community.llms import Ollama
import math

llm = Ollama(model="llama3")  # ou outro modelo que tenha

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Você é um assistente que deve delegar todos os cálculos à ferramenta chamada "Calculadora".

Regras:
- Se a pergunta for um cálculo, responda: "Usar Calculadora: <expressão>".
- Caso não seja cálculo, responda normalmente.

Exemplo:
Pergunta: Qual é 2 + 2?
Resposta: Usar Calculadora: 2 + 2
para funções de raíza quadrada você deve usar a função sqrt, por exemplo: Usar Calculadora: sqrt(16)


Agora, responda: {input}
""",
)

def calculadora(expressao: str) -> str:
    """Calculadora simples que resolve expressões matemáticas."""
    try:
        resultado = eval(expressao)
        return str(resultado)
    except Exception as e:
        return f"Erro: {e}"


chain = LLMChain(llm=llm, prompt=prompt)

def agente(input_usuario):
    resposta = chain.run(input_usuario)
    
    if "Usar Calculadora:" in resposta:
        # Extrair a expressão
        expressao = resposta.split("Usar Calculadora:")[1].strip()
        resultado = calculadora(expressao)
        return f"Resultado da Calculadora: {resultado}"
    else:
        return resposta

pergunta = "Dado ao dia ensolarado, céu azul, muitos carros passando, o asfalto ser quente, a folha das árvores serem verdes, quanto é 2 + 2?"
saida = agente(pergunta)
print(saida)


# 1. Mudar a importação e a classe para OllamaLLM
# from langchain_community.llms import Ollama # Linha antiga
from langchain_ollama import OllamaLLM       # Linha nova
from math import sqrt

# Inicializar o LLM com a nova classe
try:
    # llm = Ollama(model="gemma:2b") # Linha antiga
    llm = OllamaLLM(model="gemma:2b") # Linha nova
    # Teste rápido para ver se o LLM está acessível
    # print(llm.invoke("Diga olá"))
except Exception as e:
    print(f"Erro ao inicializar o LLM Ollama: {e}")
    print("Certifique-se de que o Ollama está em execução e o modelo 'gemma:2b' está baixado (ollama pull gemma:2b).")
    exit()

from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain # LLMChain não é mais necessária para esta forma de uso
from langchain_core.output_parsers import StrOutputParser # Útil para garantir saída de string

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
Você é um assistente que deve delegar todos os cálculos à ferramenta chamada "Calculadora".

Regras:
- Se a pergunta for um cálculo, responda APENAS: "Usar Calculadora: <expressão>". A expressão não deve conter texto adicional.
- Para funções de raiz quadrada, use a função sqrt. Exemplo: "Usar Calculadora: sqrt(16)".
- Para outras operações matemáticas (soma, subtração, multiplicação, divisão), use os operadores padrão (+, -, *, /).
- Caso não seja um cálculo, responda normalmente.

Exemplo 1:
Pergunta: Qual é 2 + 2?
Resposta: Usar Calculadora: 2 + 2

Exemplo 2:
Pergunta: Qual a raiz quadrada de 16?
Resposta: Usar Calculadora: sqrt(16)

Exemplo 3:
Pergunta: Qual a capital da França?
Resposta: A capital da França é Paris.

Agora, responda: {input}
""",
)

def calculadora(expressao: str) -> str:
    """Calculadora simples que resolve expressões matemáticas."""
    try:
        allowed_names = {"sqrt": sqrt}
        resultado = eval(expressao, {"__builtins__": {}}, allowed_names)
        return str(resultado)
    except NameError as e:
        return f"Erro: Função ou variável desconhecida na expressão '{expressao}'. Detalhe: {e}"
    except SyntaxError as e:
        return f"Erro: Sintaxe inválida na expressão '{expressao}'. Detalhe: {e}"
    except Exception as e:
        return f"Erro ao calcular '{expressao}': {e}"

# 2. Usar LCEL (LangChain Expression Language) em vez de LLMChain
# chain = LLMChain(llm=llm, prompt=prompt) # Linha antiga
chain = prompt | llm | StrOutputParser() # Linha nova (StrOutputParser garante que a saída seja uma string)

def agente(input_usuario: str) -> str:
    # Com LCEL, invoke diretamente na cadeia. A entrada ainda é um dicionário para o prompt.
    # A saída de `prompt | llm` já é a string de resposta do LLM (se llm for uma classe base LLM)
    # Adicionar StrOutputParser() explicitamente torna isso mais claro e robusto.
    resposta_llm = chain.invoke({"input": input_usuario}).strip()

    print(f"LLM disse: '{resposta_llm}'") # Para depuração

    if resposta_llm.startswith("Usar Calculadora:"):
        try:
            expressao = resposta_llm.split("Usar Calculadora:", 1)[1].strip()
            if not expressao:
                return "Erro: O LLM indicou usar a calculadora mas não forneceu uma expressão válida."
            resultado = calculadora(expressao)
            return f"Resultado da Calculadora: {resultado}"
        except IndexError:
            return f"Erro: Formato de resposta do LLM para calculadora inválido: '{resposta_llm}'"
        except Exception as e:
            return f"Erro ao processar comando da calculadora: {e}"
    else:
        return resposta_llm

# Testes
perguntas = [
    "quanto é 8 * 8",
    "qual a raiz quadrada de 25",
    "quanto é sqrt(100) / (2 + 3)",
    "Qual a capital do Brasil?",
    "quanto é 10 / 0",
    "elefante",
    "Usar Calculadora: sqrt(9)"
    "Qual o papa atual?"
]

for pergunta in perguntas:
    print(f"\nUsuário: {pergunta}")
    saida = agente(pergunta)
    print(f"Agente: {saida}")

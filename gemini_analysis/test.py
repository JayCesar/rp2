from google import genai

API_KEY = "" 


import os
from google import genai
from google.genai import types


client = genai.Client(
    api_key=API_KEY,
)

def generate(input, instrucoes):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=input,
        config=types.GenerateContentConfig(
            system_instruction=instrucoes,
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            temperature=1
        ),
    )
    print(response.text)
    return(response.text)

if __name__ == "__main__":

    input = "Vou lhe enviar uma série de redações do modelo enem, e preciso que você corrija-as, por hora, responda apenas com 'Ok.' caso tenha entendido"
    instrucoes = ""
    resposta = generate(input, instrucoes)

    if resposta == "Ok.": print("\n\nEle entendeu.")




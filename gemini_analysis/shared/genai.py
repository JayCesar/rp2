import logging

from google import genai
from google.genai import types

def init(api_key):

    try:
        client = genai.Client(
            api_key=api_key,
        )

        return client
    except Exception as e:
        logging.error(f"Erro ao inicializar cliente genai: {e}", exc_info=True)

def generate(client, input, instructions):

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=input,
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                temperature=1
            ),
        )

    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}", exc_info=True)

    return (response.text)
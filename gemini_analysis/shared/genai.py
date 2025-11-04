import logging

import httpx

from google import genai
from google.genai import types, Client

def init(api_key: str) -> Client:

    try:
        client: Client = genai.Client(
            api_key=api_key,
        )

        return client
    except Exception as e:
        logging.error(f"Erro ao inicializar cliente genai: {e}", exc_info=True)

def generate(client: Client, input: str, instructions: str = None) -> str:
    
    config_params = {
        'thinking_config': types.ThinkingConfig(thinking_budget=-1),
        'temperature': 1
    }

    if instructions is not None:
        config_params['system_instruction'] = instructions

    try:
        response: types.GenerateContentResponse = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=input,
            config=types.GenerateContentConfig(
                **config_params
            ),
        )

        return response.text

    except httpx.ReadError as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        return ""

    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}", exc_info=True)
        return ""

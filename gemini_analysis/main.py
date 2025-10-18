# Configura o logger
import logging

from . import gemini_analysis

try:
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - [%(name)s] - %(message)s'
    )

    noisy_logger = ['google.genai', 'google', 'google_genai.models', 'httpx']
    for logger in noisy_logger:
        logging.getLogger(logger).setLevel(logging.WARNING)

    gemini_analysis.run()

except KeyboardInterrupt:
    logging.info("\n\nPrograma encerrado pelo usuário\n\n")

except Exception as e:
    logging.error(f"Erro inesperado: {e}", exc_info=True)
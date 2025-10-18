import os

from ..shared.json import json_dealer

CONTROLE_DIR = os.path.join("gemini_analysis", "controle")
AVALIACOES_PATH = os.path.join(CONTROLE_DIR, "avaliacoes.json")
IDS_AVALIADOS_PATH = os.path.join(CONTROLE_DIR, "lista_avaliadas.json")

def organize_avaliacoes():
    
    lista_avaliadas = json_dealer(IDS_AVALIADOS_PATH, 'read')
    lista_avaliadas['redacoes_avaliadas'].sort()
    json_dealer(IDS_AVALIADOS_PATH, 'write', lista_avaliadas)

    avaliacoes = json_dealer(AVALIACOES_PATH, 'read')
    itens_ordenados = dict(sorted(avaliacoes.items()))
    json_dealer(AVALIACOES_PATH, 'write', itens_ordenados)

def check_avaliacoes_restantes():

    lista_avaliadas = json_dealer(IDS_AVALIADOS_PATH, 'read')
    pending = []
    
    for i in range(0, 6577):
        if i not in lista_avaliadas['redacoes_avaliadas']:
            pending.append(i)
    
    if len(pending) == 0:
        print("Todas as avaliações foram concluídas.")
        return

    print(f"Avaliações pendentes: {pending}")


if __name__ == "__main__":
    organize_avaliacoes()
    check_avaliacoes_restantes()
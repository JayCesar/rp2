# try:

#     from redacao_dict.csv_to_dict import create_redacoes_dict

#     dados_estruturados = create_redacoes_dict()
#     print(f"\nSucesso! {len(dados_estruturados)} redações foram processadas e estruturadas.\n")

#     from tqdm import tqdm
#     from feature_extraction.feature_extraction import extrair_todas_features

#     print("\nIniciando a extração de features para todas as redações (forma modular)...")

#     for redacao in tqdm(dados_estruturados, desc="Extraindo features"):

#         texto_da_redacao = redacao['texto']
        
#         vetor_de_features = extrair_todas_features(texto_da_redacao)
#         redacao['features_c1'] = vetor_de_features

#     print("\nExtração de features concluída!")

#     import numpy as np
#     from sklearn.model_selection import train_test_split

#     print("\nIniciando a preparação final para o Machine Learning...")

#     # --- PASSO 3: Criar os conjuntos de dados X e y ---

#     # X será a lista com os vetores de erro para os modelos clássicos
#     X = np.array([dado['features_c1'] for dado in dados_estruturados])

#     # y será a lista com as notas da Competência 1 (nosso "gabarito")
#     y = np.array([dado['nota_c1'] for dado in dados_estruturados])

#     # Vamos também criar uma lista separada com os textos para usar depois com LSTM/BERT
#     textos = [dado['texto'] for dado in dados_estruturados]

#     print(f"Conjuntos X (shape: {X.shape}) e y (shape: {y.shape}) criados.")


#     # --- PASSO 4: Dividir em Treino, Validação e Teste (70/15/15) ---

#     # Primeiro, separamos o conjunto de teste (15%) do resto (85%)
#     # Fazemos isso para X, y e os textos, para manter tudo alinhado
#     X_temp, X_teste, y_temp, y_teste, textos_temp, textos_teste = train_test_split(
#         X, y, textos, test_size=0.15, random_state=42
#     )

#     # Agora, dividimos o resto (85%) em treino (70%) e validação (15%)
#     # A proporção para o test_size aqui é 15% / 85%
#     X_treino, X_valid, y_treino, y_valid, textos_treino, textos_valid = train_test_split(
#         X_temp, y_temp, textos_temp, test_size=(0.15/0.85), random_state=42
#     )

#     print("\nDados divididos com sucesso!")
#     print(f"- Tamanho do conjunto de Treino: {len(X_treino)} amostras")
#     print(f"- Tamanho do conjunto de Validação: {len(X_valid)} amostras")
#     print(f"- Tamanho do conjunto de Teste: {len(X_teste)} amostras")

#     # Criar e Treinar o Modelo
#     from sklearn.linear_model import LinearRegression
#     from sklearn.metrics import mean_squared_error, cohen_kappa_score
#     import numpy as np

#     # O modelo "aprende" a relação entre os vetores de erro (X_treino) e as notas (y_treino)
#     print("Treinando o modelo de Regressão Linear...")
#     modelo_lr = LinearRegression()
#     modelo_lr.fit(X_treino, y_treino)
#     print("Modelo treinado com sucesso!")

#     # Fazer Previsões no Conjunto de Validação
#     # Usamos o conjunto de validação para ver o quão bem o modelo generaliza para dados que não viu no treino.
#     predicoes_valid_lr = modelo_lr.predict(X_valid)

#     # Arredondar as Previsões para as Notas do ENEM
#     def arredondar_para_nota_enem(predicao):
#         nota_arredondada = np.round(predicao / 40) * 40
#         return np.clip(nota_arredondada, 0, 200) # Garante que a nota fique entre 0 e 200

#     predicoes_arredondadas_lr = np.array([arredondar_para_nota_enem(p) for p in predicoes_valid_lr])


#     # Calcular as Métricas de Avaliação
#     print("\nAvaliando o modelo no conjunto de validação...")

#     # RMSE (Root Mean Squared Error)
#     rmse_lr = np.sqrt(mean_squared_error(y_valid, predicoes_valid_lr))

#     # QWK (Quadratic Weighted Kappa)
#     # Usamos as notas reais (y_valid) e as previsões arredondadas
#     qwk_lr = cohen_kappa_score(y_valid, predicoes_arredondadas_lr, weights='quadratic')

#     print(f"\nResultados da Regressão Linear (Baseline):")
#     print(f"  - RMSE: {rmse_lr:.2f}")
#     print(f"  - QWK:  {qwk_lr:.4f}")

# except KeyboardInterrupt:
#     print("\n\nEncerrando programa...")

import feature_extraction.teste as test

test.run()
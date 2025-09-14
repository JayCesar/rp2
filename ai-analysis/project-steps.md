# Passo a passo do projeto - Rascunho inicial

## Organizar os dados dos csv em dicionários python
- Usar a biblioteca pandas para ler o .csv e um loop para criar a lista de dicionários.

## Extrair 'features' da competência 1

- Ferramentas para extrair features
  - language-tool-python - Ferramenta para encontrar erros objetivos e violações de regras gramaticais claras (ortografia, concordância, crase, pontuação, etc.).
  - spaCy - Ferramenta para análises estruturais, que não são necessariamente "erros", mas sim características do texto (comprimento de sentenças, riqueza lexical, análise da sintaxe, etc.).
  - spaCy + Ajustes Manuais - Os "ajustes manuais" aqui significam a criação de listas de palavras (gírias, conectivos formais, vocabulário erudito, etc.). O spaCy entra mais para processar o texto, e a lógica manual verifica se as palavras do texto estão ou não em suas listas.

- *Features*  
  - Erros ortográficos: language-tool-python - Detecção de violação de regras.
  - Erros de pontuação: language-tool-python - Detecção de violação de regras.
  - Erros de capitalização: language-tool-python - Detecção de violação de regras.
  - Análise de sintaxe: spaCy - Análise da estrutura e dependências da frase.
  - Concordância/Conjugação verbal: language-tool-python - Detecção de violação de regras.
  - Concordância Nominal: language-tool-python - Detecção de violação de regras.
  - Voz Impessoal: spaCy - Identificação de pronomes de 1ª pessoa.
  - Regência Verbal: language-tool-python - Detecção de violação de regras.
  - Riqueza Lexical: spaCy - Cálculo sobre os lemas (forma base) das palavras.
  - Repetição: spaCy - Contagem da frequência dos lemas.
  - Uso de crase: language-tool-python - Detecção de violação de regras.
  - Uso de coloquialismo/gírias: spaCy + Listas Customizadas - Verificação contra uma lista de palavras informais.
  - Comprimento médio das sentenças: spaCy - Análise estrutural baseada na divisão precisa das sentenças.
  - Uso de Conectivos Formais: spaCy + Listas Customizadas - Verificação contra uma lista de conectivos.
  - Vocabulário Abstrato/Erudito: spaCy + Listas Customizadas - Verificação contra uma lista de palavras formais.
  - Densidade de advérbios genéricos: spaCy + Listas Customizadas - Verificação contra uma lista de advérbios.
  - Modalizadores discursivos: spaCy + Listas Customizadas - Verificação contra uma lista de modalizadores.

## Criar conjuntos de dados para Machine Learning usando os dados das features extraídos, que será usado para treinar as IAs
- Criar a matriz de features X e o vetor de notas y a partir dos dados extraídos.

## Dividir dados para treino, validação e teste - 70/15/15
- Usar a função train_test_split da biblioteca scikit-learn

## Regressão Linear
- Treinar o modelo baseline com X_treino e y_treino e avaliar seu desempenho inicial na validação

## Tokenizar para Deep Learning
- Usar Tokenizer do Keras (para LSTM) e BertTokenizer (para BERT) para converter os textos em sequências numéricas

## LSTM e BERT
- Construir as arquiteturas, treinar os modelos com os dados tokenizados e avaliá-los na validação

## Avaliar resultados usando RMSE e QWK
- Aplicar os modelos finalizados no conjunto de teste. Consolidar as métricas em uma tabela comparativa final
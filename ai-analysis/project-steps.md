# Passo a passo do projeto - Rascunho inicial

## Organizar os dados dos csv em dicionários python
- Usar a biblioteca pandas para ler o .csv e um loop para criar a lista de dicionários.

## Extrair 'features' da competência 1, como por exemplo:

- *Features*  
  - Erros ortográficos - Grafia incorreta das palavras.
  - Erros de pontuação - Uso inadequado de vírgulas, pontos, etc.
  - Erros de capitalização - Uso incorreto de maiúsculas e minúsculas.
  - Análise de sintaxe - Estrutura e organização das frases.
  - Concordância/Conjugação verbal - Relação correta entre sujeito e verbo.
  - Concordância Nominal - Acordo entre substantivos, adjetivos, etc.
  - Voz Impessoal - Uso da 3ª pessoa para manter a formalidade.
  - Regência Verbal - Relação do verbo com seus complementos (preposições).
  - Riqueza Lexical - Diversidade do vocabulário.
  - Repetição - Uso excessivo dos mesmos termos.
  - Uso de crase - Aplicação correta do acento grave (à).
  - Uso de coloquialismo/gírias - Presença de linguagem informal.
  - Comprimento médio das sentenças - Tamanho médio das frases.
  - Uso de Conectivos Formais - Variedade de palavras de transição.
  - Vocabulário Abstrato/Erudito - Uso de palavras formais e complexas.
  - Densidade de advérbios genéricos - Abuso de advérbios vagos (ex - muito, bastante).
  - Modalizadores discursivos - Palavras que indicam opinião ou certeza do autor.

- Como extrair features
  - Erros Gramaticais e Ortográficos -> language-tool-python
  - Análise Estrutural e Sintática -> spaCy
  - Análise Lexical e de Estilo -> Lógica Customizada + Listas de Palavras

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
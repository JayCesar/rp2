# Como iniciar o repo para ajudar no desenvolvimento?

- Tenha Python na sua máquina - [link de download](https://www.python.org/downloads/)
- No terminal, na pasta do repo, digite:
```
python3 -m venv venv
```

- Em seguida, baixe as libs necessárias com o seguinte comando:
```
pip install -r requirements.txt
```

- Para a lib do spaCy, é necessário um download adicional, com o seguinte comando:
```
python -m spacy download pt_core_news_lg
```

- Seu ambiente está pronto para apoiar no desenvolvimento do projeto!!

## Como rodar os modelos com FocalLoss (gamma search)

Depois de instalar as dependências, você pode rodar os modelos de classificação
com FocalLoss usando os scripts já prontos (sempre fazem busca em gamma):

### Conv1D (Component 1)

- **Features (métricas linguísticas):**

  ```bash
  uv run python ai-analysis/conv1d/conv1d_train_on_features_focal_loss.py \
    --max-samples 2000 \
    --epochs-per-gamma 5 \
    --gamma-grid "0.5,1,2,4,8"
  ```

- **Redes de embeddings (ensaios vetorizados):**

  ```bash
  uv run python ai-analysis/conv1d/conv1d_train_on_vectorized_essays_focal_loss.py \
    --max-samples 2000 \
    --epochs-per-gamma 5 \
    --gamma-grid "0.5,1,2,4,8"
  ```

### BLSTM (Component 1)

- **Features:**

  ```bash
  uv run python ai-analysis/blstm/blstm_train_on_features_focal_loss.py \
    --max-samples 2000 \
    --epochs-per-gamma 5 \
    --gamma-grid "0.5,1,2,4,8"
  ```

- **Ensaios vetorizados (BERT/BLSTM):**

  ```bash
  uv run python ai-analysis/blstm/blstm_train_on_vectorized_essays_focal_loss.py \
    --max-samples 2000 \
    --epochs-per-gamma 5 \
    --gamma-grid "0.5,1,2,4,8"
  ```

Observações rápidas:

- Se você não passar `--gamma-grid`, os scripts usam um grid padrão
  (`DEFAULT_GAMMA_VALUES`).
- Se você não passar `--epochs-per-gamma`, é usado o valor de `TrainConfig.epochs`.
- `--max-samples` reduz somente o tamanho do *train* para deixar os testes mais
  rápidos; validação e teste usam sempre o conjunto completo.

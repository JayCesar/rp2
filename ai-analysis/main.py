import pandas as pd

file_path = '../database/extended_essay-br.csv'

try:
  
  df = pd.read_csv(file_path)

  dados_estruturados = []

  for index, row in df.iterrows():
    objeto_redacao = {
      'id': index,
      'texto': row['essay'],
      'nota_c1': row['c1'],
      'nota_c2': row['c2'],
      'nota_c3': row['c3'],
      'nota_c4': row['c4'],
      'nota_c5': row['c5']
    }
    dados_estruturados.append(objeto_redacao)

  print(
    f"\nSucesso! {len(dados_estruturados)} redações foram processadas e estruturadas.\n"
  )

except FileNotFoundError:
  print(f"\nERRO: O arquivo '{file_path}' não foi encontrado.")
  print("Por favor, verifique se o caminho está correto e o arquivo existe.")

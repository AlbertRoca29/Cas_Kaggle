import pandas as pd


# url a un github d'un repositori on guardo tots els datasets que utilitzo
url = 'https://raw.githubusercontent.com/AlbertRoca29/Datasets/main/datasets/mushrooms.csv'

# url per si tens el fitxer csv baixat al directori
url = 'mushrooms.csv'

dataset = pd.read_csv(url, header=0, delimiter=',')


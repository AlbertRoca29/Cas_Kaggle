from Dataset import dataset

# La descripció dels atributs està a descripcio_atributs.png

# Visualitzar NaNs

print('\nNombre de nans per atribut\n\n',dataset.isnull().sum())


# Veure els possibles valors que pren cada variable

print('\n\nValors que pot pendre cada variable\n',
      '%-27s %-48s %s'%('Nom','Possibles valors', 'nombre'),
      '-'*85)
_=[print('%-27s %-50s %d'%(col,dataset[col].unique(),len(dataset[col].unique()))) for col in dataset]
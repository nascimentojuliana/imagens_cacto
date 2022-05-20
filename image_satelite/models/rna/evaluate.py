import pandas as pd
from odonto.models.rna.neural_network import RNA
from odonto.models.DataGenerator import DataGenerator

methods = ['RGB']

mode = 'xception'

df = pd.read_csv('')

df_test = df[(df.use == 'test')].reset_index(drop=True)

for method in methods:

    path_model = ''

    model = RNA(path_model=path_model, dimension=299)

    dict_scores = model.evaluate(df=df_test, batch_size=1, method=method) 

    with open((""), "w") as json_file:
        json_file.write(str(dict_scores))

    print('finalizado {}'.format(method))

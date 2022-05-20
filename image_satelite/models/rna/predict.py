import pandas as pd
from odonto.models.rna.neural_network import RNA

methods = ['RGB']

mode = 'xception'

df = pd.read_csv('')

df_test = df[(df.use == 'test')].reset_index(drop=True)

for method in methods:

    path_model = ''

    model = RNA(path_model=path_model, dimension=299)

    submission_results = model.predict(df=df, batch_size=1, method=method) 

    submission_results.to_csv('')

    print('finalizado {}'.format(method))

import pandas as pd
from image_satelite.models.rna.neural_network import RNA
from sklearn.model_selection import train_test_split
from image_satelite.models.DataGenerator import DataGenerator

methods = ['RGB_NDVI']

df = pd.read_csv('aerial-cactus-identification/train.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop(
'label', axis=1), df['label'], test_size=0.2, stratify=df[['label']], random_state=42)

X_train = pd.concat((X_train, y_train), axis=1)
X_train['use'] = 'train'
X_test = pd.concat((X_test, y_test), axis=1)
X_test['use'] = 'test'

df = pd.concat((X_train, X_test))

df_test = df[(df.use == 'test')].reset_index(drop=True)

for method in methods:

    path_model = 'models/model_6_canais.h5'

    model = RNA(path_model=path_model, dimension=299)

    dict_scores = model.evaluate(df=df_test, batch_size=1, method=method) 

    with open(("models/evaluate_6_canais.json"), "w") as json_file:
        json_file.write(str(dict_scores))

    print('finalizado {}'.format(method))

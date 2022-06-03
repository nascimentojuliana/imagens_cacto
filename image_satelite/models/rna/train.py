import pandas as pd
from image_satelite.models.rna.neural_network import RNA
from sklearn.model_selection import train_test_split


def train():
    methods = ['RGB'] 

    model_base = 'xception_3_channels.h5'

    df = pd.read_csv('aerial-cactus-identification/train.csv')

    X_train, X_validation, y_train, y_validation = train_test_split(df.drop(
    'label', axis=1), df['label'], test_size=0.2, stratify=df[['label']], random_state=42)

    X_train = pd.concat((X_train, y_train), axis=1)
    X_train['use'] = 'train'
    X_val = pd.concat((X_validation, y_validation), axis=1)
    X_val['use'] = 'validate'

    df = pd.concat((X_train, X_val))

    df_train = df[(df.use == 'train')].reset_index(drop=True)

    df_validate = df[(df.use == 'validate')].reset_index(drop=True)

    for method in methods:

        model = RNA(dimension = 299)

        model = model.fit(df_test=None, 
    			    	  df_validate=df_validate, 
    			    	  df_train=df_train,
    			    	  method = method,
    			    	  model_base = model_base,
    			    	  batch_size=32,
    			    	  epochs=10)

        model_json = model.to_json()
        with open('modelo_3_bandas.json', "w") as json_file:
        	json_file.write(model_json)

        model.save('modelo_6_bandas.h5')
        print("Saved model to disk")


if __name__ == '__main__':

    train()

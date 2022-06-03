import os, sys
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import load_model 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from image_satelite.models.DataGenerator import DataGenerator

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize


class RNA():
	def __init__(self,path_model=None, dimension=None):

		self.path_model = path_model
		self.dimension = dimension
		    
		if self.path_model:
			self.model = self.load_model()

	def load_model(self):
		model = load_model('{}'.format(self.path_model))
		return model

	def fit(self, df_test, df_validate, df_train, method, model_base, epochs=1000, batch_size=24):

		print('###########################{}##########################'.format(method))

		# create the base pre-trained models
		#model_base = Xception(weights='imagenet', include_top=False)
		model_base = load_model(model_base)

		checkpoint_filepath = 'model.h5'

		callback = [EarlyStopping(monitor='loss', patience=20), 
					ModelCheckpoint(filepath=checkpoint_filepath, 
					monitor="val_loss", verbose=0, 
					save_best_only=True, save_weights_only=False,
					mode="auto", save_freq="epoch", options=None)]


		csv_logger = CSVLogger('logs.csv', append=True, separator=';')

		# add a global spatial average pooling layer
		x = model_base.output
		x = GlobalAveragePooling2D()(x)
		# let's add a fully-connected layer
		x = Dense(1024, activation='relu')(x)
		# and a logistic layer -- let's say we have 200 classes
		outputs = Dense(2, activation='softmax')(x)

		# this is the model we will train
		model = Model(inputs=model_base.input, outputs=outputs)

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		data_train = DataGenerator(df=df_train, batch_size=batch_size, method=method, dimension=self.dimension, shuffle=True) 

		data_validate = DataGenerator(df=df_validate, batch_size=batch_size, method=method, dimension=self.dimension, shuffle=True) 

		model.fit(data_train, epochs=epochs, validation_data=(data_validate), callbacks=[callback, csv_logger])

		os.remove(checkpoint_filepath)

		return model

	def predict(self, df, batch_size, method):
		data = DataGenerator(df=df, batch_size=batch_size, method=method, dimension=self.dimension, shuffle=False) 
		predictions = self.model.predict(data) 

		scores = pd.DataFrame(predictions, columns = ['normal', 'alterada'])
		scores_y = scores[['alterada']].values

		submission_results =self.seleciona_classe_threshold(scores)

		submission_results.insert(0, 'image', df['image_name'])

		submission_results.insert(0, 'label', df['label'])

		return submission_results, scores_y

	def seleciona_classe_threshold(self, df):
		df.columns = ['normal',  'alterada']
		df['predito'] = ''
		for index, row in df.iterrows():
			normal = row.normal
			alterada = row.alterada
			if alterada >= 0.9:
				df.at[index, 'predito'] = 1
			if alterada < 0.9:
				df.at[index,'predito'] = 0
		return df


	def evaluate(self, df, batch_size, method):

		submission_results, scores_y = self.predict(df, batch_size, method)

		submission_results.insert(1, 'real', df['label'])

		submission_results['real'] = submission_results[['real']].applymap(lambda x: int(x))
		
		Y = label_binarize(submission_results['real'], classes=[0, 1])

		auc = roc_auc_score(Y, scores_y)

		# n_classes = Y.shape[1]

		# fpr = dict()
		# tpr = dict()
		# roc_auc = dict()
		# for i in range(n_classes):
		# 	fpr[i], tpr[i], _ = roc_curve(Y[:, i], scores_y[:, i])
		# 	roc_auc[i] = auc(fpr[i], tpr[i])

		# fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), scores_y.ravel())
		# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		submission_results['real'] = submission_results[['real']].applymap(lambda x: str(x))
		submission_results['predito'] = submission_results[['predito']].applymap(lambda x: str(x))

		matrix = confusion_matrix(submission_results['real'], submission_results['predito'])

		accuracy = accuracy_score(submission_results['real'], submission_results['predito'], normalize=False)

		recall = recall_score(submission_results['real'], submission_results['predito'], average='macro')

		f1 = f1_score(submission_results['real'], submission_results['predito'], average='macro')

		precision = precision_score(submission_results['real'], submission_results['predito'], average='macro')

		dict_result = {'accuracy_score': accuracy,
						'recall_score': recall,
						'f1_score': f1,
						'confusion_matrix': matrix,
						'precision': precision,
						'auc': auc}

		return dict_result

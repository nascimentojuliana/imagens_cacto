import tensorflow as tf
import math
import cv2
from image_satelite.models.rna.pre_processing import PreProcessing
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications import imagenet_utils
import numpy as np
from tensorflow import keras
import random
from tqdm import tqdm

class DataGeneratorPredict(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size, dimension, shuffle=None, method=None): 
        self.dimension = dimension
        self.pre_processing = PreProcessing(dimension)
        self.df = df # your pandas dataframe 
        self.bsz = batch_size # batch size
        self.method = method # shuffle when in train method 
        
        self.im_list = self.df['image_name'].tolist()
        self.method = method 
        self.shuffle = shuffle
        self.on_epoch_end()

    def pre_process(self, im):
        try:
            if self.method == 'RGB':
                imagem = image.load_img(im, target_size=(self.dimension , self.dimension))
                imagem = image.img_to_array(imagem)
                imagem = np.expand_dims(imagem, axis=0)
                imagem = imagenet_utils.preprocess_input(imagem, mode='tf')[0]

            if self.method == 'ELA':
                imagem = image.load_img(im, target_size=(self.dimension , self.dimension))
                imagem = image.img_to_array(imagem)
                imagem = np.expand_dims(imagem, axis=0)
                imagem = imagenet_utils.preprocess_input(imagem, mode='tf')[0]
                imagem = self.pre_processing.transform_ela(imagem)
        except:
            pass

        return imagem


    def __len__(self): # compute number of batches to yield 
        return int(math.ceil(len(self.df) / float(self.bsz))) 

    def on_epoch_end(self): # Shuffles indexes after each epoch if in training method 
        self.indexes = range(len(self.im_list)) 
        if self.shuffle == True:
            self.indexes = random.sample(self.indexes, k=len(self.indexes)) 

        self.indexes = np.arange(len(self.im_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_batch_features(self, idx): # Fetch a batch of inputs 
        indexes = self.indexes[idx * self.bsz: (1 + idx) * self.bsz]
        x = [(self.pre_process(self.im_list[k])) for k in indexes]
        return np.array(x)

    def __getitem__(self, idx): 
        batch_x = self.get_batch_features(idx) #[lxaxcx24]
        return (batch_x)

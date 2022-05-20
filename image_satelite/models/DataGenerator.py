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
from PIL import Image
import io

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size, dimension, shuffle=None, method=None): 
        self.pre_processing = PreProcessing(dimension)
        self.dimension = dimension
        self.df = df # your pandas dataframe 
        self.df.columns = ['image_name', 'label', 'use']
        self.bsz = batch_size # batch size
        self.method = method # shuffle when in train method 
        # Take labels and a list of image locations in memory 
        
        self.im_list = self.df['image_name'].tolist()
        self.methbucketod = method 
        self.shuffle = shuffle
        self.on_epoch_end()

    def preprocess_images(self, im):

        if self.method == 'RGB':
            imagem = Image.open(im).convert('RGB').resize((self.dimension, self.dimension), Image.ANTIALIAS)
            imagem = image.img_to_array(imagem)
            imagem = np.expand_dims(imagem, axis=0)
            imagem = imagenet_utils.preprocess_input(imagem, mode='tf')[0]

        if self.method == 'RGB_NDVI':
            imagem_original = Image.open(im).convert('RGB').resize((self.dimension, self.dimension), Image.ANTIALIAS)
            imagem = image.img_to_array(imagem_original)
            imagem = np.expand_dims(imagem, axis=0)
            imagem = imagenet_utils.preprocess_input(imagem, mode='tf')[0]
            imagem_ndvi = preprocess_ndvi(imagem_original)
            result = np.concatenate((imagem, imagem_ndvi), axis=2)

        return result

    def __len__(self): # compute number of batches to yield 
        return int(math.ceil(len(self.df) / float(self.bsz))) 

    def on_epoch_end(self): # Shuffles indexes after each epoch if in training method 
        self.indexes = range(len(self.im_list)) 
        if self.shuffle == True:
            self.indexes = random.sample(self.indexes, k=len(self.indexes)) 

        self.indexes = np.arange(len(self.im_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_batch_labels(self, idx): # Fetch a batch of labels
        labels = self.df['label'].values  
        indexes = self.indexes[idx * self.bsz: (1 + idx) * self.bsz]
        y = [labels[k] for k in indexes]
        return keras.utils.to_categorical(y, num_classes=2)

    def get_batch_features(self, idx): # Fetch a batch of inputs 
        indexes = self.indexes[idx * self.bsz: (1 + idx) * self.bsz]
        x = [(self.preprocess_images(self.im_list[k])) for k in indexes]
        return np.array(x)

    def __getitem__(self, idx): 
        batch_x = self.get_batch_features(idx) #[lxaxcx24]
        batch_y = self.get_batch_labels(idx) 
        return (batch_x, batch_y)
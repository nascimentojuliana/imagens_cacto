import cv2
import numpy as np
import PIL
from PIL import Image

class PreProcessing():
	def __init__(self, dimension):
		self.dimension = dimension

	def transform(self, image):
		img = Image.open(image)
		r = img.getchannel(0)
		g = img.getchannel(1)
		r_g = PIL.ImageChops.difference(r,g)
		r_plus_g = PIL.ImageChops.add(r, g)
		r_g = np.array(r_g)
		r_plus_g = np.array(r_plus_g)
		result = r_g/((r_plus_g.astype('float')+1)/256)
		result = result*(result < 255)+255*np.ones(np.shape(result))*(result > 255)
		c = result.astype('uint8')
		imgOut = Image.fromarray(result).convert('RGB').resize((self.dimension, self.dimension), Image.ANTIALIAS)

		return imgOut

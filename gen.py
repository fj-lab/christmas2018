import os, sys
import tifffile
import numpy as np
import keras
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
np.random.seed(42)

# Save generated images
def save_generated_images(generated_images):
   
    for i in range(len(generated_images)):
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        name = './generated/{}.jpg'.format(str(i))
        plt.imsave(name, image.astype(np.uint8))
    #plt.imshow(image.astype(np.uint8))
    #plt.show()

# Training
def test(model_path, generate_size):
    # Load model
    model = load_model(model_path)
    
    # Generate images
    noise = np.random.normal(0, 1, size=(generate_size, ) + (1, 1, 100))
    generated_images = model.predict(noise)
    
    save_generated_images(generated_images)

  
def main():
    model_path = './models/generator_epoch30.hdf5'
    generate_size = 64
    test(model_path, generate_size)


if __name__ == "__main__":
    main()

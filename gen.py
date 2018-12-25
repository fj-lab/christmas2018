import os, sys
import tifffile
import numpy as np
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
np.random.seed(42)

# Save generated images
def save_generated_images(generated_images):
   
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(4, 8)
    gs.update(wspace=0, hspace=0)

    for i in range(len(generated_images)):
        ax = plt.subplot(gs[i])
        ax.set_aspect('equal')

        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        #name = './generated/{}.jpg'.format(str(i))
        #plt.imsave(name, image.astype(np.uint8))
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.savefig('gs.jpg', bbox_inches='tight', pad_inches=0)
     
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
    model_path = './models/generator_epoch200.hdf5'
    generate_size = 32
    test(model_path, generate_size)


if __name__ == "__main__":
    main()

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def compress_image(image_path, k):
    # Open the image
    image = Image.open(image_path)
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Reshape the image array to a 2D array of pixels
    pixels = image_array.reshape(-1, 3)
    
    # Create a KMeans instance
    kmeans = KMeans(n_clusters=k)
    
    # Fit the KMeans model to the pixel data
    kmeans.fit(pixels)
    
    # Get the cluster labels for each pixel
    labels = kmeans.predict(pixels)
    
    # Replace each pixel with its corresponding centroid value
    compressed_pixels = kmeans.cluster_centers_[labels]
    
    # Reshape the compressed pixels back to the original image shape
    compressed_image_array = compressed_pixels.reshape(image_array.shape)
    
    # Create a PIL image from the compressed image array
    compressed_image = Image.fromarray(np.uint8(compressed_image_array))
    
    # Save the compressed image
    compressed_image.save('compressed_image.jpg')

if __name__ == '__main__':
    compress_image('1.jpeg', k=16)
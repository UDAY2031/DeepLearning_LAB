import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Read the image
img = cv2.imread('s.png')

# Convert to grayscale
ig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
e = cv2.equalizeHist(ig)

# Apply thresholding
_, t = cv2.threshold(ig, 127, 255, cv2.THRESH_BINARY)

# Detect edges
ed = cv2.Canny(ig, 100, 200)

# Flip image horizontally
flip = cv2.flip(ig, 1)

# Apply morphological operation
m = cv2.morphologyEx(ig, cv2.MORPH_CLOSE, np.ones((5, 5)))

# List of images and titles
imgs = [img, e, t, ed, flip, m]
titles = ['original', 'equalized', 'threshold', 'edge', 'flip', 'morphology']

# Show all grayscale results
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Prepare image for augmentation
x = np.expand_dims(img_to_array(img), axis=0)

# Define augmentation techniques
gen = ImageDataGenerator(rotation_range=30, zoom_range=0.2,
                         horizontal_flip=True, vertical_flip=True)

# Show original and augmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 5, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

# Display 4 augmented images
for i, batch in enumerate(gen.flow(x, batch_size=1)):
    plt.subplot(1, 5, i + 2)
    plt.imshow(batch[0].astype('uint8'))
    plt.title(f"Aug{i + 1}")
    plt.axis('off')
    if i == 3:
        break

plt.tight_layout()
plt.show()

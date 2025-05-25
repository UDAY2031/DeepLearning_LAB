import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('sample.png', 0)

# Apply operations
eq = cv2.equalizeHist(img)
_, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
edge = cv2.Canny(img, 100, 200)
flip = cv2.flip(img, 1)
morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# Show results
names = ['Original', 'Equalized', 'Threshold', 'Edges', 'Flip', 'Morph']
imgs = [img, eq, th, edge, flip, morph]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(names[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

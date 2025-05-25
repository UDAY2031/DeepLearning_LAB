import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, :]
    return img

content = load_img('c.png')
style = load_img('s.png')
style = tf.image.resize(style, content.shape[1:3])

model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
s = model(content, style)[0]

plt.imshow(s[0])
plt.axis('off')
plt.show()

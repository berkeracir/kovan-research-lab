import tensorflow as tf
from glob import glob
import os

lst = []

for fn in glob("Images/*/*"):
    with tf.Graph().as_default():
        image_contents = tf.read_file(fn)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        init_op = tf.initialize_all_tables()
        with tf.Session() as sess:
            try:
                sess.run(init_op)
                tmp = sess.run(image)
            except:
                lst.append(fn)

for i in lst:
    os.remove(i)
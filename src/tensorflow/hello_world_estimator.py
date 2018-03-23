# -*- coding: utf-8 -*-

import tensorflow as tf


def compute_area(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]

    s = (a + b + c) * 0.5
    areasq = s * (s - a) * (s - b) * (s - c)
    return tf.sqrt(areasq)


with tf.Session() as sess:
    area = compute_area(
        tf.constant([
            [5.0, 3.0, 7.1],
            [2.3, 4.1, 4.8]
        ]))

    result = sess.run(area)
    print(result)

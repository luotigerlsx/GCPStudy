# -*- coding: utf-8 -*-

import tensorflow as tf

"""
Once you have built a Dataset to represent your input data, 
the next step is to create an Iterator to access elements from that dataset.
"""

"""
One-shot iterator

the simplest form of iterator, which only supports iterating once through a dataset, 
with no need for explicit initialization. One-shot iterators handle almost all of the cases 
that the existing queue-based input pipelines support, but they do not support parameterization.
"""

# dataset = tf.data.Dataset.range(100)
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     value_holder = []
#     for i in range(100):
#         value = sess.run(next_element)
#         value_holder.append(value)
#     print(value_holder)

"""
Initializable iterator

requires you to run an explicit iterator.initializer operation before using it. 
In exchange for this inconvenience, it enables you to parameterize the definition of the dataset, 
using one or more tf.placeholder() tensors that can be fed when you initialize the iterator.
"""

# max_value = tf.placeholder(tf.int64, shape=[])
# dataset = tf.data.Dataset.range(max_value)
# print(dataset)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     sess.run(iterator.initializer, feed_dict={max_value: 10})
#     value_holder = []
#     for i in range(10):
#         value = sess.run(next_element)
#         value_holder.append(value)
#     print(value_holder)


"""
Reinitializable iterator

can be initialized from multiple different Dataset objects
"""

training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))

validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(
    output_types=training_dataset.output_types,
    output_shapes=training_dataset.output_shapes
)

next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
with tf.Session() as sess:
    for _ in range(2):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(100):
            sess.run(next_element)

        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
            sess.run(next_element)

"""
Feedable iterator:

used together with tf.placeholder to select what Iterator to use in each call to tf.Session.run
"""
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

with tf.Session() as sess:

    # The `Iterator.string_handle()` method returns a low_level_api that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    for _ in range(200):
        sess.run(next_element, feed_dict={handle: training_handle})

    # Run one pass over the validation dataset.
    sess.run(validation_iterator.initializer)
    for _ in range(50):
        sess.run(next_element, feed_dict={handle: validation_handle})

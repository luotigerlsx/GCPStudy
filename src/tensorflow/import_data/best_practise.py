import tensorflow as tf

files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")

dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers, block_length=10))

dataset = dataset.map(..., num_parallel_calls=FLAGS.num_parallel_calls)

if shuffle:
    dataset = dataset.shuffle(buffer_size=batch_size * 10)

dataset = dataset.repeat(FLAGS.num_epochs)
dataset = dataset.batch(32)
dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
    sess.run(iterator.initializer)
    while True:
        try:
            sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break

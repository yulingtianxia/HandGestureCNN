import os

import tensorflow as tf


IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160

# Global constants describing the HandGesture data set.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100


def read_image(queue):
    reader = tf.WholeFileReader()
    filename, content = reader.read(queue, name='read_image')
    filename_split = tf.string_split([filename], delimiter='/')
    label_id = tf.string_to_number(filename_split.values[1], out_type=tf.int32) - 1
    label_id = tf.reshape(label_id, [1])
    img_tensor = tf.image.decode_png(
        content,
        dtype=tf.uint8,
        channels=3,
        name='img_decode')

    return img_tensor, label_id


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for HandGesture training using the Reader ops.

  Args:
    data_dir: Path to the HandGesture data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  pattern = os.path.join(data_dir, '*/*.png')
  filenames = tf.train.match_filenames_once(pattern, name='list_files')

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  image, label = read_image(filename_queue)

  resized_image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

  reshaped_image = tf.cast(resized_image, tf.float32)

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(reshaped_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d HandGesture images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for HandGesture evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the HandGesture data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  pattern = os.path.join(data_dir, '*/*.png')
  filenames = tf.train.match_filenames_once(pattern, name='list_files')
  if not eval_data:
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  image, label = read_image(filename_queue)
  reshaped_image = tf.cast(image, tf.float32)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(reshaped_image)

  # Set the shapes of tensors.
  float_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
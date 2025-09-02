import tensorflow as tf
tf.disable_v2_behavior()   # turn off TF2.x behaviors (eager execution, etc.)

# Check GPU availability
print("=== GPU Check ===")
print("Is GPU Available?:", tf.compat.v1.test.is_gpu_available())
print("GPU Devices:", tf.compat.v1.config.list_physical_devices('GPU'))

# Define a simple matrix multiplication to run on GPU if available
with tf.device('/GPU:0'):
    a = tf.compat.v1.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.compat.v1.constant([[2.0, 0.0], [0.0, 2.0]])
    c = tf.compat.v1.matmul(a, b)

with tf.Session() as sess:
    result = sess.run(c)
    print("Matrix Multiplication Result:\n", result)

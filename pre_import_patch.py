# simple_tf_patch.py
"""
Simple TensorFlow 1.x compatibility patch.
Import this BEFORE importing tfsnippet.
"""


def patch_tensorflow():
    """Patch TensorFlow for TF1 compatibility"""
    print("Applying TensorFlow 1.x compatibility patch...")

    import tensorflow as tf
    import tensorflow.compat.v1 as tf_v1

    # Disable TF2 behavior
    tf_v1.disable_v2_behavior()
    tf_v1.disable_eager_execution()

    # Monkey patch the main tensorflow module
    tf.GraphKeys = tf_v1.GraphKeys
    tf.Session = tf_v1.Session
    tf.placeholder = tf_v1.placeholder
    tf.variable_scope = tf_v1.variable_scope
    tf.get_variable_scope = tf_v1.get_variable_scope
    tf.name_scope = tf_v1.name_scope
    tf.global_variables_initializer = tf_v1.global_variables_initializer
    tf.train = tf_v1.train
    tf.layers = tf_v1.layers

    # Handle contrib
    try:
        import tf_slim as slim
        tf.contrib = slim
        tf.contrib.framework = slim
    except ImportError:
        class MockContrib:
            class framework:
                @staticmethod
                def add_arg_scope(func):
                    return func

                @staticmethod
                def arg_scope(*args, **kwargs):
                    return lambda func: func

        tf.contrib = MockContrib()

    print(f"TensorFlow {tf.__version__} patched for TF1 compatibility")

    return tf


# Apply the patch immediately when this module is imported
patch_tensorflow()
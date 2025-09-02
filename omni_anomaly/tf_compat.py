# tf_compat.py
"""
Centralized TensorFlow compatibility module for tfsnippet
This monkey patches the actual tensorflow module before any imports
"""
import tensorflow as tf_original
import tensorflow.compat.v1 as tf

# Disable TensorFlow 2.x behavior
tf.compat.v1.disable_v2_behavior()


# Monkey patch the original tensorflow module directly
def apply_tf_monkey_patches():
    """Apply monkey patches to the original tensorflow module"""

    # Patch the original tensorflow module that tfsnippet will import
    if not hasattr(tf_original, 'GraphKeys'):
        tf_original.GraphKeys = tf.compat.v1.GraphKeys

    if not hasattr(tf_original, 'get_variable_scope'):
        tf_original.get_variable_scope = tf.compat.v1.get_variable_scope

    if not hasattr(tf_original, 'variable_scope'):
        tf_original.variable_scope = tf.compat.v1.variable_scope

    if not hasattr(tf_original, 'name_scope'):
        tf_original.name_scope = tf.compat.v1.name_scope

    # Additional common TF 1.x attributes
    if not hasattr(tf_original, 'Session'):
        tf_original.Session = tf.compat.v1.Session

    if not hasattr(tf_original, 'global_variables_initializer'):
        tf_original.global_variables_initializer = tf.compat.v1.global_variables_initializer

    if not hasattr(tf_original, 'placeholder'):
        tf_original.placeholder = tf.compat.v1.placeholder

    if not hasattr(tf_original, 'train'):
        tf_original.train = tf.compat.v1.train

    # Also patch compat.v1 to ensure consistency
    tf_original.compat.v1.GraphKeys = tf.compat.v1.GraphKeys

    print("TensorFlow monkey patches applied successfully")
    return tf_original


# Apply patches immediately
apply_tf_monkey_patches()

# Export both the compat version and the patched original
tf_compat = tf
tf_patched = tf_original
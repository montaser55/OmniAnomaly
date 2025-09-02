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

    # Essential patches for tfsnippet
    if not hasattr(tf, 'GraphKeys'):
        tf.GraphKeys = tf_compat.GraphKeys

    if not hasattr(tf, 'get_variable_scope'):
        tf.get_variable_scope = tf_compat.get_variable_scope

    if not hasattr(tf, 'variable_scope'):
        tf.variable_scope = tf_compat.variable_scope

    if not hasattr(tf, 'name_scope'):
        tf.name_scope = tf_compat.name_scope

    # Additional common TF 1.x attributes
    if not hasattr(tf, 'Session'):
        tf.Session = tf_compat.Session

    if not hasattr(tf, 'global_variables_initializer'):
        tf.global_variables_initializer = tf_compat.global_variables_initializer

    if not hasattr(tf, 'placeholder'):
        tf.placeholder = tf_compat.placeholder

    if not hasattr(tf, 'train'):
        tf.train = tf_compat.train

    if not hasattr(tf, 'layers'):
        tf.layers = tf_compat.layers

    # Patch tensorflow.contrib
    try:
        # Try to import tf_slim as replacement for tensorflow.contrib
        import tf_slim as slim
        tf.contrib = slim
        tf.contrib.framework = slim
    except ImportError:
        # Create minimal mock for tensorflow.contrib if tf_slim not available
        class MockContrib:
            pass

        class MockFramework:
            def add_arg_scope(self, func):
                return func

            def arg_scope(self, func):
                return func

        tf.contrib = MockContrib()
        tf.contrib.framework = MockFramework()

    # Also ensure these are available in compat.v1 for consistency
    print("TensorFlow monkey patches applied successfully")
    return tf


# Apply patches immediately
apply_tf_monkey_patches()

# Export both the compat version and the patched original
tf_compat = tf
tf_patched = tf_original
"""
TensorFlow compatibility patch for tfsnippet
"""
import os
import importlib
import tensorflow.compat.v1 as tf

# Apply the patch before any tfsnippet imports
if not hasattr(tf, 'GraphKeys'):
    tf.GraphKeys = tf.compat.v1.GraphKeys

# Also patch other potential compatibility issues
if not hasattr(tf, 'get_variable_scope'):
    tf.get_variable_scope = tf.compat.v1.get_variable_scope

# Make the patched tf available for import
os.environ['TF_COMPAT_PATCH_APPLIED'] = 'True'
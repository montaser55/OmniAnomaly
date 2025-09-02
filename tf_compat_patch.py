# tf_compat.py
"""
Centralized TensorFlow compatibility module for tfsnippet
"""
import tensorflow.compat.v1 as tf

# Apply all necessary compatibility patches
def setup_tf_compatibility():
    """Setup TensorFlow compatibility for tfsnippet and other TF 1.x libraries"""
    
    # Essential patches for tfsnippet
    if not hasattr(tf, 'GraphKeys'):
        tf.GraphKeys = tf.compat.v1.GraphKeys
    
    if not hasattr(tf, 'get_variable_scope'):
        tf.get_variable_scope = tf.compat.v1.get_variable_scope
    
    if not hasattr(tf, 'variable_scope'):
        tf.variable_scope = tf.compat.v1.variable_scope
    
    if not hasattr(tf, 'name_scope'):
        tf.name_scope = tf.compat.v1.name_scope
    
    # Additional common TF 1.x attributes that might be needed
    if not hasattr(tf, 'Session'):
        tf.Session = tf.compat.v1.Session
    
    if not hasattr(tf, 'global_variables_initializer'):
        tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
    
    if not hasattr(tf, 'placeholder'):
        tf.placeholder = tf.compat.v1.placeholder
    
    if not hasattr(tf, 'train'):
        tf.train = tf.compat.v1.train
    
    print("TensorFlow compatibility patches applied successfully")
    return tf

# Apply patches immediately when this module is imported
setup_tf_compatibility()

# Export the patched tf module
tf_compat = tf
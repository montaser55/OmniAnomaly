
"""
TensorFlow 2.9.3 + Python 3.8 compatibility patch for tfsnippet
Optimized for your specific setup.
"""
import sys
import types
import warnings


def patch_for_tf():
    """Patch specifically for TensorFlow 2.9.3"""

    print("Applying TensorFlow 2.9.3 compatibility patch...")

    # Create contrib modules in sys.modules BEFORE tensorflow import
    framework_module = types.ModuleType('tensorflow.contrib.framework')

    def add_arg_scope(func):
        """Mock add_arg_scope - just returns the function unchanged"""
        return func

    def arg_scope(*args, **kwargs):
        """Mock arg_scope context manager"""

        class ArgScopeContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def decorator(func):
            return func

        if args and callable(args[0]):
            # Called as decorator directly
            return decorator(args[0])
        else:
            # Called as context manager
            return ArgScopeContext()

    # Add functions to framework module
    framework_module.add_arg_scope = add_arg_scope
    framework_module.arg_scope = arg_scope

    # Create contrib module
    contrib_module = types.ModuleType('tensorflow.contrib')
    contrib_module.framework = framework_module

    # Install in sys.modules
    sys.modules['tensorflow.contrib'] = contrib_module
    sys.modules['tensorflow.contrib.framework'] = framework_module

    # Now safely import tensorflow
    import tensorflow as tf
    import tensorflow.compat.v1 as tf_v1

    print(f"TensorFlow version: {tf.__version__}")

    # Configure for TF1 behavior
    tf_v1.disable_v2_behavior()
    tf_v1.disable_eager_execution()

    # Suppress warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    tf.get_logger().setLevel('ERROR')

    # Try to enhance with tf-slim
    try:
        import tf_slim as slim
        print("Found tf-slim, enhancing contrib module...")

        # Add slim functionality to contrib
        for attr_name in dir(slim):
            if not attr_name.startswith('_') and not hasattr(contrib_module, attr_name):
                setattr(contrib_module, attr_name, getattr(slim, attr_name))

        # Ensure framework is still our mock (slim might override)
        contrib_module.framework.add_arg_scope = add_arg_scope
        contrib_module.framework.arg_scope = arg_scope

    except ImportError:
        print("tf-slim not installed, using basic contrib mock")

    # Link contrib to tensorflow
    tf.contrib = contrib_module
    tf_v1.contrib = contrib_module

    # Apply TF1 compatibility patches
    tf.GraphKeys = tf_v1.GraphKeys
    tf.Session = tf_v1.Session
    tf.placeholder = tf_v1.placeholder
    tf.variable_scope = tf_v1.variable_scope
    tf.get_variable_scope = tf_v1.get_variable_scope
    tf.name_scope = tf_v1.name_scope
    tf.global_variables_initializer = tf_v1.global_variables_initializer
    tf.train = tf_v1.train
    tf.layers = tf_v1.layers
    tf.get_variable = tf_v1.get_variable
    tf.random_normal = tf_v1.random_normal
    tf.random_uniform = tf_v1.random_uniform

    # Configure session for TF 2.9.3
    try:
        config = tf_v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        tf_v1.keras.backend.set_session(tf_v1.Session(config=config))
        print("GPU configuration applied")
    except Exception as e:
        print(f"Note: GPU configuration skipped: {e}")

    print("✅ TensorFlow 2.9.3 successfully patched for tfsnippet compatibility")
    print(f"✅ contrib.framework.add_arg_scope available: {hasattr(tf.contrib.framework, 'add_arg_scope')}")

    return tf


# Apply patch immediately
patch_for_tf()
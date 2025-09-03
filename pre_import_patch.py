# pre_import_patch.py
"""
Force TensorFlow patching before ANY imports happen.
This must be imported as the VERY FIRST thing in your main script.
"""
import sys
import os
import importlib

# Store original modules to detect what's already loaded
_original_modules = set(sys.modules.keys())


class TFPreImportPatcher:
    def __init__(self):
        self.patched = False

    def find_spec(self, fullname, path, target=None):
        # Patch TensorFlow when it's about to be imported
        if not self.patched and ('tensorflow' in fullname or 'tfsnippet' in fullname):
            self._patch_tensorflow_globally()
        return None

    def _patch_tensorflow_globally(self):
        if self.patched:
            return

        print("Patching TensorFlow globally for TF1 compatibility...")

        # Import TensorFlow modules
        import tensorflow as tf
        import tensorflow.compat.v1 as tf_v1

        # Force TF1 behavior
        tf_v1.disable_v2_behavior()
        tf_v1.disable_eager_execution()

        # MONKEY PATCH THE ACTUAL tensorflow MODULE
        # This ensures tfsnippet will see the patched version
        tf.GraphKeys = tf_v1.GraphKeys
        tf.Session = tf_v1.Session
        tf.placeholder = tf_v1.placeholder
        tf.variable_scope = tf_v1.variable_scope
        tf.get_variable_scope = tf_v1.get_variable_scope
        tf.name_scope = tf_v1.name_scope
        tf.global_variables_initializer = tf_v1.global_variables_initializer
        tf.train = tf_v1.train
        tf.layers = tf_v1.layers

        # Also patch compat.v1 to ensure consistency
        tf.compat.v1.GraphKeys = tf_v1.GraphKeys
        tf.compat.v1.get_variable_scope = tf_v1.get_variable_scope

        # Handle tensorflow.contrib replacement
        try:
            import tf_slim as slim
            tf.contrib = slim
            tf.contrib.framework = slim
            print("Using tf_slim for tensorflow.contrib replacement")
        except ImportError:
            print("Warning: tf_slim not installed, creating minimal contrib mock")

            class MockContrib:
                framework = type('MockFramework', (), {
                    'add_arg_scope': staticmethod(lambda func: func),
                    'arg_scope': staticmethod(lambda *args, **kwargs: lambda func: func)
                })()

            tf.contrib = MockContrib()

        # Configure GPU for optimal performance
        gpu_config = tf_v1.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        tf_v1.keras.backend.set_session(tf_v1.Session(config=gpu_config))

        self.patched = True
        print(f"TensorFlow {tf.__version__} patched globally for TF1 compatibility")
        print("GraphKeys available:", hasattr(tf, 'GraphKeys'))
        print("GLOBAL_VARIABLES available:", hasattr(tf.GraphKeys, 'GLOBAL_VARIABLES'))


# Install the import hook
sys.meta_path.insert(0, TFPreImportPatcher())


# Force reload any already imported modules that might use tensorflow
def _reload_tf_dependent_modules():
    for module_name in list(sys.modules.keys()):
        if module_name not in _original_modules:
            if any(keyword in module_name for keyword in ['tensorflow', 'tfsnippet']):
                try:
                    del sys.modules[module_name]
                    print(f"Reloaded module: {module_name}")
                except:
                    pass


_reload_tf_dependent_modules()
# runtime_tf_fix.py
"""
Runtime fix for TensorFlow missing functions.
Import this after your comprehensive_tf_patch and before any ML libraries.
"""


def apply_runtime_tf_fixes():
    """Apply runtime fixes for common TensorFlow issues"""

    import tensorflow as tf
    import sys

    print("🔧 Applying runtime TensorFlow fixes...")

    # Get tensorflow module
    tf_module = sys.modules.get('tensorflow')
    if tf_module is None:
        print("❌ TensorFlow module not found")
        return False

    # List of critical math functions that need to be available
    math_functions = [
        'log', 'exp', 'sqrt', 'square', 'abs', 'sign', 'round', 'ceil', 'floor',
        'maximum', 'minimum', 'pow', 'mod', 'logical_and', 'logical_or', 'logical_not',
        'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
        'reduce_sum', 'reduce_mean', 'reduce_max', 'reduce_min', 'reduce_prod',
        'reduce_all', 'reduce_any', 'argmax', 'argmin'
    ]

    fixed_count = 0

    for func_name in math_functions:
        if not hasattr(tf, func_name):
            try:
                # Get the function from tf.math
                math_func = getattr(tf.math, func_name)

                # Apply using multiple methods
                setattr(tf, func_name, math_func)
                setattr(tf_module, func_name, math_func)
                tf_module.__dict__[func_name] = math_func

                fixed_count += 1
                print(f"✅ Fixed tf.{func_name}")

            except AttributeError:
                print(f"⚠️  Could not fix tf.{func_name} - not found in tf.math")
        else:
            print(f"✅ tf.{func_name} already available")

    print(f"🎉 Runtime fixes applied: {fixed_count} functions fixed")

    # Test tf.log specifically
    try:
        test_tensor = tf.constant([1.0, 2.0])
        result = tf.log(test_tensor)
        print(f"✅ tf.log test successful: {result}")
        return True
    except Exception as e:
        print(f"❌ tf.log test failed: {e}")
        return False


def monkeypatch_tf_log():
    """Last resort: monkey patch tf.log into the TensorFlow module"""

    import tensorflow as tf
    import sys

    if hasattr(tf, 'log'):
        print("✅ tf.log already exists")
        return True

    try:
        # Direct module manipulation
        tf_module = sys.modules['tensorflow']

        # Create a wrapper function that calls tf.math.log
        def tf_log_wrapper(*args, **kwargs):
            return tf.math.log(*args, **kwargs)

        # Assign it multiple ways
        tf.log = tf_log_wrapper
        tf_module.log = tf_log_wrapper
        tf_module.__dict__['log'] = tf_log_wrapper
        sys.modules['tensorflow'].log = tf_log_wrapper

        # Test it
        test_result = tf.log(tf.constant([1.0, 2.0]))
        print(f"🎉 Monkey patch successful: tf.log = {test_result}")
        return True

    except Exception as e:
        print(f"❌ Monkey patch failed: {e}")
        return False


if __name__ == "__main__":
    # Can be run standalone to test
    apply_runtime_tf_fixes()
    if not hasattr(__import__('tensorflow'), 'log'):
        print("Trying monkey patch...")
        monkeypatch_tf_log()
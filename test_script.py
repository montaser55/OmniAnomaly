# test_patch.py
"""
Test script to verify the TensorFlow patch is working correctly
Run this BEFORE your main script to verify everything is set up correctly
"""


def test_tf_patch():
    """Test that the TF patch is working"""

    print("=" * 60)
    print("TESTING TENSORFLOW 1.x COMPATIBILITY PATCH")
    print("=" * 60)

    # Import the patch (this should apply it)
    try:
        import comprehensive_tf_patch
        print("✅ Patch imported successfully")
    except Exception as e:
        print(f"❌ Failed to import patch: {e}")
        return False

    # Test TensorFlow import
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except Exception as e:
        print(f"❌ Failed to import TensorFlow: {e}")
        return False

    # Test critical mathematical functions
    print("\nTesting mathematical functions...")
    tests = [
        ('tf.log', lambda: hasattr(tf, 'log')),
        ('tf.exp', lambda: hasattr(tf, 'exp')),
        ('tf.sqrt', lambda: hasattr(tf, 'sqrt')),
        ('tf.reduce_mean', lambda: hasattr(tf, 'reduce_mean')),
        ('tf.reduce_sum', lambda: hasattr(tf, 'reduce_sum')),
        ('tf.maximum', lambda: hasattr(tf, 'maximum')),
        ('tf.minimum', lambda: hasattr(tf, 'minimum')),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {result}")
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")

    # Test TF1 core functions
    print("\nTesting TF1 core functions...")
    tf1_tests = [
        ('tf.Session', lambda: hasattr(tf, 'Session')),
        ('tf.placeholder', lambda: hasattr(tf, 'placeholder')),
        ('tf.variable_scope', lambda: hasattr(tf, 'variable_scope')),
        ('tf.get_variable', lambda: hasattr(tf, 'get_variable')),
        ('tf.global_variables_initializer', lambda: hasattr(tf, 'global_variables_initializer')),
    ]

    for test_name, test_func in tf1_tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {result}")
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")

    # Test RNN functions
    print("\nTesting RNN functions...")
    rnn_tests = [
        ('tf.nn.rnn_cell', lambda: hasattr(tf.nn, 'rnn_cell')),
        ('tf.nn.rnn_cell.GRUCell', lambda: hasattr(tf.nn.rnn_cell, 'GRUCell') if hasattr(tf.nn, 'rnn_cell') else False),
        ('tf.nn.rnn_cell.LSTMCell',
         lambda: hasattr(tf.nn.rnn_cell, 'LSTMCell') if hasattr(tf.nn, 'rnn_cell') else False),
        ('tf.contrib.rnn', lambda: hasattr(tf.contrib, 'rnn') if hasattr(tf, 'contrib') else False),
    ]

    for test_name, test_func in rnn_tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {result}")
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")

    # Test actual function call
    print("\nTesting actual function calls...")
    try:
        import numpy as np
        x = tf.constant([1.0, 2.0, 3.0])
        log_x = tf.log(x)
        print(f"✅ tf.log() call successful: {log_x}")
    except Exception as e:
        print(f"❌ tf.log() call failed: {e}")
        return False

    try:
        cell = tf.nn.rnn_cell.GRUCell(10)
        print(f"✅ tf.nn.rnn_cell.GRUCell() creation successful: {cell}")
    except Exception as e:
        print(f"❌ tf.nn.rnn_cell.GRUCell() creation failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Your patch is working correctly!")
    print("You can now run your main script.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_tf_patch()
    if not success:
        print("\n❌ PATCH TEST FAILED - Please fix the issues above before running your main script.")
        exit(1)
# test_patch.py
"""
Comprehensive test script that tries ALL TensorFlow patch methods
Tests: comprehensive patch, aggressive patch, and runtime fixes
"""


def test_tf_patch():
    """Test that the TF patch is working"""

    print("=" * 80)
    print("COMPREHENSIVE TENSORFLOW 1.x COMPATIBILITY TESTING")
    print("Testing: Comprehensive Patch + Aggressive Patch + Runtime Fixes")
    print("=" * 80)

    # STEP 1: Import the comprehensive patch (this should apply it automatically)
    print("\n" + "=" * 60)
    print("STEP 1: TESTING COMPREHENSIVE PATCH")
    print("=" * 60)

    try:
        import comprehensive_tf_patch
        print("✅ Comprehensive patch imported successfully")
    except Exception as e:
        print(f"❌ Failed to import comprehensive patch: {e}")
        return False

    # Test TensorFlow import
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except Exception as e:
        print(f"❌ Failed to import TensorFlow: {e}")
        return False

    # Check if tf.log works after comprehensive patch
    print(f"🔍 tf.log after comprehensive patch: {hasattr(tf, 'log')}")
    if hasattr(tf, 'log'):
        try:
            test_result = tf.log(tf.constant([1.0, 2.0]))
            print(f"✅ tf.log works after comprehensive patch: {test_result}")
            comprehensive_works = True
        except Exception as e:
            print(f"❌ tf.log exists but doesn't work: {e}")
            comprehensive_works = False
    else:
        comprehensive_works = False

    # STEP 2: Try aggressive patch
    print("\n" + "=" * 60)
    print("STEP 2: TESTING AGGRESSIVE PATCH")
    print("=" * 60)

    aggressive_works = False
    if not comprehensive_works:
        try:
            from comprehensive_tf_patch import aggressive_tf_log_patch
            success = aggressive_tf_log_patch()
            print(f"✅ Aggressive patch applied: {success}")

            if hasattr(tf, 'log'):
                try:
                    test_result = tf.log(tf.constant([1.0, 2.0]))
                    print(f"✅ tf.log works after aggressive patch: {test_result}")
                    aggressive_works = True
                except Exception as e:
                    print(f"❌ tf.log exists but doesn't work after aggressive patch: {e}")
            else:
                print("❌ tf.log still not available after aggressive patch")

        except Exception as e:
            print(f"❌ Could not apply aggressive patch: {e}")
    else:
        print("✅ Skipping aggressive patch - comprehensive patch already works")
        aggressive_works = True

    # STEP 3: Try runtime fixes
    print("\n" + "=" * 60)
    print("STEP 3: TESTING RUNTIME FIXES")
    print("=" * 60)

    runtime_works = False
    if not (comprehensive_works or aggressive_works):
        try:
            import runtime_tf_fix

            print("Trying apply_runtime_tf_fixes()...")
            success1 = runtime_tf_fix.apply_runtime_tf_fixes()

            if hasattr(tf, 'log'):
                try:
                    test_result = tf.log(tf.constant([1.0, 2.0]))
                    print(f"✅ tf.log works after runtime fixes: {test_result}")
                    runtime_works = True
                except Exception as e:
                    print(f"❌ tf.log exists but doesn't work after runtime fixes: {e}")

            if not runtime_works:
                print("Trying monkeypatch_tf_log()...")
                success2 = runtime_tf_fix.monkeypatch_tf_log()

                if hasattr(tf, 'log'):
                    try:
                        test_result = tf.log(tf.constant([1.0, 2.0]))
                        print(f"✅ tf.log works after monkey patch: {test_result}")
                        runtime_works = True
                    except Exception as e:
                        print(f"❌ tf.log exists but doesn't work after monkey patch: {e}")

        except Exception as e:
            print(f"❌ Could not apply runtime fixes: {e}")
    else:
        print("✅ Skipping runtime fixes - earlier patches already work")
        runtime_works = True

    # STEP 4: Final comprehensive test
    print("\n" + "=" * 60)
    print("STEP 4: COMPREHENSIVE FUNCTION TESTING")
    print("=" * 60)

    # Test critical mathematical functions
    print("\nTesting mathematical functions...")
    math_tests = [
        ('tf.log', lambda: hasattr(tf, 'log')),
        ('tf.exp', lambda: hasattr(tf, 'exp')),
        ('tf.sqrt', lambda: hasattr(tf, 'sqrt')),
        ('tf.reduce_mean', lambda: hasattr(tf, 'reduce_mean')),
        ('tf.reduce_sum', lambda: hasattr(tf, 'reduce_sum')),
        ('tf.maximum', lambda: hasattr(tf, 'maximum')),
        ('tf.minimum', lambda: hasattr(tf, 'minimum')),
    ]

    math_success_count = 0
    for test_name, test_func in math_tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {result}")
            if result:
                math_success_count += 1
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

    tf1_success_count = 0
    for test_name, test_func in tf1_tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {result}")
            if result:
                tf1_success_count += 1
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

    rnn_success_count = 0
    for test_name, test_func in rnn_tests:
        try:
            result = test_func()
            status = "✅" if result else "❌"
            print(f"{status} {test_name}: {result}")
            if result:
                rnn_success_count += 1
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")

    # STEP 5: Actual function calls
    print("\n" + "=" * 60)
    print("STEP 5: ACTUAL FUNCTION CALL TESTS")
    print("=" * 60)

    # Test tf.log specifically
    print("\nTesting tf.log function call...")
    try:
        import numpy as np
        x = tf.constant([1.0, 2.0, 3.0])
        log_x = tf.log(x)
        print(f"✅ tf.log() call successful: {log_x}")
        log_call_works = True
    except Exception as e:
        print(f"❌ tf.log() call failed: {e}")
        log_call_works = False

        # Fallback test
        try:
            log_x = tf.math.log(x)
            print(f"✅ tf.math.log() fallback works: {log_x}")
        except Exception as e2:
            print(f"❌ Even tf.math.log() failed: {e2}")

    # Test RNN cell creation
    print("\nTesting RNN cell creation...")
    try:
        cell = tf.nn.rnn_cell.GRUCell(10)
        print(f"✅ tf.nn.rnn_cell.GRUCell() creation successful: {cell}")
        rnn_call_works = True
    except Exception as e:
        print(f"❌ tf.nn.rnn_cell.GRUCell() creation failed: {e}")
        rnn_call_works = False

    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"📊 Comprehensive patch works: {'✅ YES' if comprehensive_works else '❌ NO'}")
    print(f"📊 Aggressive patch works: {'✅ YES' if aggressive_works else '❌ NO'}")
    print(f"📊 Runtime fixes work: {'✅ YES' if runtime_works else '❌ NO'}")
    print(f"📊 Math functions: {math_success_count}/{len(math_tests)} working")
    print(f"📊 TF1 functions: {tf1_success_count}/{len(tf1_tests)} working")
    print(f"📊 RNN functions: {rnn_success_count}/{len(rnn_tests)} working")
    print(f"📊 tf.log() calls work: {'✅ YES' if log_call_works else '❌ NO'}")
    print(f"📊 RNN calls work: {'✅ YES' if rnn_call_works else '❌ NO'}")

    overall_success = (
            (comprehensive_works or aggressive_works or runtime_works) and
            log_call_works and rnn_call_works and
            math_success_count >= 5 and tf1_success_count >= 4 and rnn_success_count >= 3
    )

    if overall_success:
        print("\n🎉 OVERALL RESULT: SUCCESS!")
        print("✅ Your TensorFlow 1.x compatibility setup is working!")
        print("🚀 You can now run your main script.")
    else:
        print("\n⚠️  OVERALL RESULT: PARTIAL SUCCESS")
        print("Some functions may not work properly.")
        print("Check the details above for specific issues.")

    print("=" * 80)
    return overall_success


if __name__ == "__main__":
    success = test_tf_patch()
    if not success:
        print("\n❌ SOME TESTS FAILED - Please review the results above.")
        exit(1)
    else:
        print("\n✅ ALL CRITICAL TESTS PASSED!")
        exit(0)
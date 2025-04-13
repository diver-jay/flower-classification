import tensorflow as tf
print("TensorFlow 버전:", tf.__version__)
print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    # GPU 테스트
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(c)
    print("GPU 테스트 성공!")
else:
    print("GPU를 사용할 수 없습니다.")

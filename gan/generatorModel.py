from tensorflow.keras.layers import Reshape, LeakyReLU, Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import Model

class GeneratorModel(Model):
    def __init__(self, *args, **kwargs):
        super(GeneratorModel, self).__init__(*args, **kwargs)

        self.d1 = Dense(256, input_shape=(100, ), use_bias=False)
        self.b1 = BatchNormalization()
        self.l1 = LeakyReLU()

        self.d2 = Dense(256, use_bias=False)
        self.b2 = BatchNormalization()
        self.l2 = LeakyReLU()

        self.d3 = Dense(28 * 28 * 1, use_bias=False, activation="tanh")
        self.b3 = BatchNormalization()

        self.r1 = Reshape((28,28,1))

    def call(self, x):
        x = self.d1(x)
        x = self.b1(x)
        x = self.l1(x)
        
        x = self.d2(x)
        x = self.b2(x)
        x = self.l2(x)

        x = self.d3(x)
        x = self.b3(x)

        y = self.r1(x)

        return y

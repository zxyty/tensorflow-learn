from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras import Model

class DiscriminatorModel(Model):
    def __init__(self, *args, **kwargs):
        super(DiscriminatorModel, self).__init__(*args, **kwargs)

        self.f1 = Flatten()

        self.d1 = Dense(512, use_bias=False)
        self.b1 = BatchNormalization()
        self.l1 = LeakyReLU()

        self.d2 = Dense(512, use_bias=False)
        self.b2 = BatchNormalization()
        self.l2 = LeakyReLU()

        self.d3 = Dense(1)
    
    def call(self, x):
        x = self.f1(x)
        
        x = self.d1(x)
        x = self.b1(x)
        x = self.l1(x)

        x = self.d2(x)
        x = self.b2(x)
        x = self.l2(x)

        y = self.d3(x)

        return y

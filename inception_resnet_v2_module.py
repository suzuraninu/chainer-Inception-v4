import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from common_module import ConvBN


class Stem(chainer.Chain):
    def __init__(self, ch):
        super(Stem, self).__init__()
        initialW = initializers.GlorotNormal()
        with self.init_scope():
            self.convbn1 = ConvBN(ch, 32, ksize=3, stride=2,
                                  initialW=initialW, pad=0)
            self.convbn2 = ConvBN(32, 32, ksize=3, stride=1,
                                  initialW=initialW, pad=0)
            self.convbn3 = ConvBN(32, 64, ksize=3, stride=1,
                                  initialW=initialW, pad=1)
            self.convbn4 = ConvBN(64, 96, ksize=3, stride=2,
                                  initialW=initialW, pad=0)


            self.convbn5_1x1_1 = ConvBN(160, 64, ksize=1, stride=1,
                                        initialW=initialW, pad=0)            

            self.convbn5_7x1 = ConvBN(64, 64, ksize=(7, 1), stride=1,
                                      initialW=initialW, pad=(3, 0))
            
            self.convbn5_1x7 = ConvBN(64, 64, ksize=(1, 7), stride=1,
                                      initialW=initialW, pad=(0, 3))
            self.convbn5_3x3_1 = ConvBN(64, 96, ksize=3, stride=1,
                                        initialW=initialW, pad=0)
            
            self.convbn5_1x1_2 = ConvBN(160, 64, ksize=1, stride=1,
                                        initialW=initialW, pad=0)
            self.convbn5_3x3_2 = ConvBN(64, 96, ksize=3, stride=1,
                                        initialW=initialW, pad=0)

            self.convbn6 = ConvBN(192, 192, ksize=3, stride=2,
                                  initialW=initialW, pad=0)            

    def __call__(self, x):
        h = F.relu(self.convbn1(x))
        h = F.relu(self.convbn2(h))
        h = F.relu(self.convbn3(h))
        
        h1 = F.max_pooling_2d(h, ksize=3, stride=2)
        h2 = F.relu(self.convbn4(h))
        h = F.concat((h1, h2), axis=1)

        h1 = F.relu(self.convbn5_1x1_1(h))
        h1 = F.relu(self.convbn5_7x1(h1))
        h1 = F.relu(self.convbn5_1x7(h1))
        h1 = F.relu(self.convbn5_3x3_1(h1))

        h2 = F.relu(self.convbn5_1x1_2(h))
        h2 = F.relu(self.convbn5_3x3_2(h2))
        
        h = F.concat((h1, h2), axis=1)

        h1 = F.relu(self.convbn6(h))
        h2 = F.max_pooling_2d(h, ksize=3, stride=2, pad=0)

        h = F.concat((h1, h2), axis=1)
        
        return h


class Inception_Resnet_A(chainer.Chain):
    def __init__(self, ch, scale=1.0):
        super(Inception_Resnet_A, self).__init__()
        self.scale = scale
        initialW = initializers.GlorotNormal()
        with self.init_scope():
            self.convbn1 = ConvBN(ch, 32, ksize=1, stride=1,
                                  initialW=initialW, pad=0)
            
            self.convbn2_1 = ConvBN(ch, 32, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn2_2 = ConvBN(32, 32, ksize=3, stride=1,
                                    initialW=initialW, pad=1)

            self.convbn3_1 = ConvBN(ch, 32, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn3_2 = ConvBN(32, 48, ksize=3, stride=1,
                                    initialW=initialW, pad=1)
            self.convbn3_3 = ConvBN(48, 64, ksize=3, stride=1,
                                    initialW=initialW, pad=1)

            self.conv4 = L.Convolution2D(128, ch, ksize=1, stride=1,
                                         initialW=initialW, pad=0)

    def __call__(self, x):
        h1 = F.relu(self.convbn1(x))

        h2 = F.relu(self.convbn2_1(x))
        h2 = F.relu(self.convbn2_2(h2))

        h3 = F.relu(self.convbn3_1(x))
        h3 = F.relu(self.convbn3_2(h3))
        h3 = F.relu(self.convbn3_3(h3))

        h = F.concat((h1, h2, h3), axis=1)
        h = self.conv4(h)
        
        return x + h * self.scale


class Inception_Resnet_B(chainer.Chain):
    def __init__(self, ch, scale=1.0):
        super(Inception_Resnet_B, self).__init__()
        self.scale = scale
        initialW = initializers.GlorotNormal()
        with self.init_scope():
            self.convbn1 = ConvBN(ch, 192, ksize=1, stride=1,
                                  initialW=initialW, pad=0)
            
            self.convbn2_1 = ConvBN(ch, 128, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn2_2 = ConvBN(128, 160, ksize=(1, 7), stride=1,
                                    initialW=initialW, pad=(0, 3))

            self.convbn2_3 = ConvBN(160, 192, ksize=(7, 1), stride=1,
                                    initialW=initialW, pad=(3, 0))
            
            self.conv3 = L.Convolution2D(384, ch, ksize=1, stride=1,
                                         initialW=initialW, pad=0)

    def __call__(self, x):
        h1 = F.relu(self.convbn1(x))

        h2 = F.relu(self.convbn2_1(x))
        h2 = F.relu(self.convbn2_2(h2))
        h2 = F.relu(self.convbn2_3(h2))

        h = F.concat((h1, h2), axis=1)
        h = self.conv3(h)
        
        return x + h * self.scale


class Reduction_B(chainer.Chain):
    def __init__(self, ch):
        super(Reduction_B, self).__init__()
        initialW = initializers.GlorotNormal()
        with self.init_scope():
            self.convbn2_1 = ConvBN(ch, 256, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn2_2 = ConvBN(256, 384, ksize=3, stride=2,
                                    initialW=initialW, pad=0)

            self.convbn3_1 = ConvBN(ch, 256, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn3_2 = ConvBN(256, 288, ksize=3, stride=2,
                                    initialW=initialW, pad=0)
            
            self.convbn4_1 = ConvBN(ch, 256, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn4_2 = ConvBN(256, 288, ksize=3, stride=1,
                                    initialW=initialW, pad=1)
            
            self.convbn4_3 = ConvBN(288, 320, ksize=3, stride=2,
                                    initialW=initialW, pad=0)
            
    def __call__(self, x):
        h1 = F.max_pooling_2d(x, ksize=3, stride=2)

        h2 = F.relu(self.convbn2_1(x))
        h2 = F.relu(self.convbn2_2(h2))

        h3 = F.relu(self.convbn3_1(x))
        h3 = F.relu(self.convbn3_2(h3))

        h4 = F.relu(self.convbn4_1(x))
        h4 = F.relu(self.convbn4_2(h4))
        h4 = F.relu(self.convbn4_3(h4))

        h = F.concat((h1, h2, h3, h4), axis=1)

        return h


class Inception_Resnet_C(chainer.Chain):
    def __init__(self, ch, scale=1.0):
        super(Inception_Resnet_C, self).__init__()
        self.scale = 1.0
        initialW = initializers.GlorotNormal()
        with self.init_scope():
            self.convbn1 = ConvBN(ch, 192, ksize=1, stride=1,
                                  initialW=initialW, pad=0)
            
            self.convbn2_1 = ConvBN(ch, 192, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn2_2 = ConvBN(192, 224, ksize=(1, 3), stride=1,
                                    initialW=initialW, pad=(0, 1))
            self.convbn2_3 = ConvBN(224, 256, ksize=(3, 1), stride=1,
                                    initialW=initialW, pad=(1, 0))
            
            self.conv3 = L.Convolution2D(448, ch, ksize=1, stride=1,
                                         initialW=initialW, pad=0)

    def __call__(self, x):
        h1 = F.relu(self.convbn1(x))

        h2 = F.relu(self.convbn2_1(x))
        h2 = F.relu(self.convbn2_2(h2))
        h2 = F.relu(self.convbn2_3(h2))

        h = F.concat((h1, h2), axis=1)
        h = self.conv3(h)
        
        return x + h * self.scale
    

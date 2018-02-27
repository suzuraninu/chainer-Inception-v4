import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

class ConvBN(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1,
                 pad=0, nobias=False, initialW=None, initial_bias=None,
                 decay=0.9997, eps=0.001):
        """ Convolution and BatchNormalization
        """
        super(ConvBN, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize=ksize,
                                        stride=stride, pad=pad, nobias=nobias,
                                        initialW=initialW, initial_bias=initial_bias)
            self.bn = L.BatchNormalization(out_channels, decay=decay, eps=eps)

    def __call__(self, x):
        return self.bn(self.conv(x))


class Reduction_A(chainer.Chain):
    def __init__(self, ch, k, l, m, n):
        super(Reduction_A, self).__init__()
        initialW = initializers.GlorotNormal()
        with self.init_scope():
            self.convbn2_1 = ConvBN(ch, n, ksize=3, stride=2,
                                    initialW=initialW, pad=0)

            self.convbn3_1 = ConvBN(ch, k, ksize=1, stride=1,
                                    initialW=initialW, pad=0)
            self.convbn3_2 = ConvBN(k, l, ksize=3, stride=1,
                                    initialW=initialW, pad=1)
            self.convbn3_3 = ConvBN(l, m, ksize=3, stride=2,
                                    initialW=initialW, pad=0)
            
    def __call__(self, x):
        h1 = F.max_pooling_2d(x, ksize=3, stride=2)

        h2 = F.relu(self.convbn2_1(x))

        h3 = F.relu(self.convbn3_1(x))
        h3 = F.relu(self.convbn3_2(h3))
        h3 = F.relu(self.convbn3_3(h3))

        h = F.concat((h1, h2, h3), axis=1)
        
        return h

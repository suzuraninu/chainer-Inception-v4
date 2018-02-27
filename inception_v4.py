import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from common_module import Reduction_A
from inception_v4_module import *


class Inception_v4(chainer.Chain):
    def __init__(self, n_classes):
        self.in_size = (299, 299)
        self.n_classes = n_classes
        super(Inception_v4, self).__init__()
        with self.init_scope():
            self.Stem = Stem(3)
            
            for i in range(1, 4):
                self.add_link('Inc_A_{}'.format(i), Inception_A(384))
            
            self.Red_A = Reduction_A(384, 192, 224, 256, 384)
            
            for i in range(1, 8):
                self.add_link('Inc_B_{}'.format(i), Inception_B(1024))
            
            self.Red_B = Reduction_B(1024)
            
            self.Inc_C = Inception_C(1536)
            for i in range(1, 4):
                self.add_link('Inc_C_{}'.format(i), Inception_C(1536))
            
            self.fc = L.Linear(1536, self.n_classes)

    def forward(self, x):
        h = self.Stem(x)

        for i in range(1, 4):
            h = self['Inc_A_{}'.format(i)](h)
        h = self.Red_A(h)
        for i in range(1, 8):
            h = self['Inc_B_{}'.format(i)](h)
        h = self.Red_B(h)
        for i in range(1, 4):
            h = self['Inc_C_{}'.format(i)](h)

        h = F.average(h, axis=(2, 3))

        h = F.dropout(h, 0.8)
        h = self.fc(h)
        
        return h

    def __call__(self, x, t):
        h = self.forward(x)
        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        
        chainer.report({
            'loss': loss,
            'accuracy': accuracy
        }, self)
        
        return loss
    

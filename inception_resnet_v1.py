import chainer
import chainer.functions as F
import chainer.links as L

from common_module import Reduction_A, ConvBN
from inception_resnet_v1_module import *


class Inception_Resnet_v1(chainer.Chain):
    def __init__(self, n_classes):
        self.in_size = (299, 299)
        self.n_classes = n_classes
        super(Inception_Resnet_v1, self).__init__()
        with self.init_scope():
            self.Stem = Stem(3)
            
            for i in range(1, 6):
                self.add_link('Inc_A_{}'.format(i), Inception_Resnet_A(256, scale=0.17))
            
            self.Red_A = Reduction_A(256, 192, 192, 256, 384)

            for i in range(1, 11):
                self.add_link('Inc_B_{}'.format(i), Inception_Resnet_B(896, scale=0.1))
            
            self.Red_B = Reduction_B(896)

            for i in range(1, 5):
                self.add_link('Inc_C_{}'.format(i), Inception_Resnet_C(1792, scale=0.2))
            self.add_link('Inc_C_5', Inception_Resnet_C(1792, scale=1.))
                
            self.convbn = ConvBN(1792, 1536, ksize=1, stride=1, pad=0)
            
            self.fc = L.Linear(1536, self.n_classes)

    def forward(self, x):
        h = self.Stem(x)
        for i in range(1, 6):
            h = self['Inc_A_{}'.format(i)](h)
        h = self.Red_A(h)
        for i in range(1, 11):
            h = self['Inc_B_{}'.format(i)](h)
        h = self.Red_B(h)
        for i in range(1, 6):
            h = self['Inc_C_{}'.format(i)](h)
        h = F.relu(self.convbn(h))
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
    

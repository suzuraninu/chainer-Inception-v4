import chainer
import chainer.functions as F
import chainer.links as L

from common_module import Reduction_A, ConvBN
from inception_resnet_v2_module import *

class Inception_Resnet_v2(chainer.Chain):
    def __init__(self, n_classes):
        self.in_size = (299, 299)
        self.n_classes = n_classes
        super(Inception_Resnet_v2, self).__init__()
        with self.init_scope():
            self.Stem = Stem(3)

            for i in range(1, 11):
                self.add_link('Inc_A_{}'.format(i), Inception_Resnet_A(384, scale=0.17))
            
            self.Red_A = Reduction_A(384, 256, 256, 384, 384)

            for i in range(1, 21):
                self.add_link('Inc_B_{}'.format(i), Inception_Resnet_B(1152, scale=0.1))
            
            self.Red_B = Reduction_B(1152)

            for i in range(1, 10):
                self.add_link('Inc_C_{}'.format(i), Inception_Resnet_C(2144, scale=0.2))
            self.add_link('Inc_C_10', Inception_Resnet_C(2144, scale=1.0))

            self.convbn = ConvBN(2144, 1536, ksize=1, stride=1, pad=0)
            self.fc = L.Linear(1536, self.n_classes)

    def forward(self, x):
        h = self.Stem(x)
        for i in range(1, 11):
            h = self['Inc_A_{}'.format(i)](h)
        h = self.Red_A(h)
        for i in range(1, 21):
            h = self['Inc_B_{}'.format(i)](h)
        h = self.Red_B(h)
        for i in range(1, 11):
            h = self['Inc_C_{}'.format(i)](h)
        h = F.relu(self.convbn(h))
        h = F.average(h, axis=(2, 3))
        h = F.dropout(h, 0.8)
        h = self.fc(h)
        return h

    def predict_10_crops(self, x):
        sides = self.in_size[0]
        p = int(x.shape[2] / 2. - sides / 2.)
        indices = x.shape[0]
            
        top_left = x[:, :, :sides, :sides]
        top_right = x[:, :, :sides, -sides:]
        bot_left = x[:, :, -sides:, :sides]
        bot_right = x[:, :, -sides:, -sides:]
        center = x[:, :, p: p + sides, p: p + sides]
        
        top_left_r = top_left[:, :, :, ::-1]
        top_right_r = top_right[:, :, :, ::-1]
        bot_left_r = bot_left[:, :, :, ::-1]
        bot_right_r = bot_right[:, :, :, ::-1]
        center_r = center[:, :, :, ::-1]
        inputs = F.concat((top_left, top_right, bot_left, bot_right,
                           center, top_left_r, top_right_r, bot_left_r,
                           bot_right_r, center_r), axis=0)
        outputs = F.softmax(self.forward(inputs))
        # outputs = F.split_axis(outputs, indices, axis=0)
        outputs = [outputs[i::indices, :] for i in range(indices)]
        return [F.average(output, axis=0) for output in outputs]

    
    def __call__(self, x, t):
        h = self.forward(x)
        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        
        chainer.report({
            'loss': loss,
            'accuracy': accuracy
        }, self)
        
        return loss
    
if __name__ == '__main__':
    import numpy as np
    x = np.random.rand(2, 3, 320, 320).astype(np.float32)
    model = Inception_Resnet_v2(55)
    predicts = model.predict_10_crops(x)
    for p in predicts:
        p = p.Data
        print(p, p.sum())

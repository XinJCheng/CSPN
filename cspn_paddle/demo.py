'''
    3d CSPN demo module
'''

import numpy as np
import paddle.fluid as fluid


class CSPN3D(object):
    '''
        3d cspn class
    '''
    def __init__(self, prop_kernel=3, prop_step=24):
        self.prop_kernel = prop_kernel
        self.prop_step = prop_step

    def cspn3d(self, guide, feat):
        '''
            3d cspn module
        '''
        guide = fluid.layers.abs(guide)
        guide_num = guide.shape[1] // feat.shape[1]
        layer_name = 'cspn3d_affinity_propagate'
        if feat.shape[1] > 1:
            cspn_feat = list()
            for channel_ind in range(feat.shape[1]):
                slice_guide = fluid.layers.slice(
                    guide, axes=[1], starts=[channel_ind * guide_num],
                    ends=[(channel_ind + 1) * guide_num])
                normalizer = fluid.layers.reduce_sum(slice_guide, dim=1, keep_dim=True)
                normalizer = fluid.layers.expand(normalizer, expand_times=[1, guide_num, 1, 1, 1])
                slice_guide = fluid.layers.elementwise_div(slice_guide, normalizer)
                slice_feat = fluid.layers.slice(feat, axes=[1], starts=[channel_ind],
                                                ends=[channel_ind + 1])
                for _ in range(self.prop_step):
                    # gate_weight: normalized guidance, shared across all the channels
                    slice_feat = fluid.layers.affinity_propagate(
                        slice_feat, gate_weight=slice_guide, kernel_size=self.prop_kernel,
                        name=layer_name)
                cspn_feat.append(slice_feat)
            cspn_feat = fluid.layers.concat(cspn_feat, axis=1)
        else:
            normalizer = fluid.layers.reduce_sum(guide, dim=1, keep_dim=True)
            normalizer = fluid.layers.expand(normalizer, expand_times=[1, guide_num, 1, 1, 1])
            guide = fluid.layers.elementwise_div(guide, normalizer)
            for _ in range(self.prop_step):
                feat = fluid.layers.affinity_propagate(
                    feat, gate_weight=guide, kernel_size=self.prop_kernel, name=layer_name)
            cspn_feat = feat
        return cspn_feat

    def demo(self, batch_size=3, iter_num=20, feat_chan=1, map_shape=(48, 64, 128)):
        '''
            func to run demo
        '''
        # define net
        assert len(map_shape) == 3, '3d map shape is required'
        guide_chan = self.prop_kernel ** 3 - 1
        guide_shape = [feat_chan * guide_chan, *map_shape]
        guide = fluid.layers.data(name='guide', shape=guide_shape)
        feat_shape = [feat_chan, *map_shape]
        feat = fluid.layers.data(name='feat', shape=feat_shape, stop_gradient=False)
        cspn_feat = self.cspn3d(guide, feat)
        output = fluid.layers.reduce_mean(cspn_feat)
        # define optim
        optim = fluid.optimizer.AdamOptimizer()
        optim.minimize(output)
        # initialize param
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        # train
        for i in range(iter_num):
            guide_data = np.random.rand(batch_size, *guide_shape).astype(np.float32)
            feat_data = np.random.rand(batch_size, *feat_shape).astype(np.float32)
            outs = exe.run(feed={'guide': guide_data, 'feat': feat_data}, fetch_list=[output.name])
            print('iter={:02}  out={:.4f}'.format(i, outs[0][0]))

if __name__ == '__main__':
    MODULE = CSPN3D()
    MODULE.demo()

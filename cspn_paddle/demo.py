'''
    CSPN module demo
'''

import argparse
import numpy as np
import paddle.fluid as fluid


class CSPN():
    '''
        cspn module
    '''
    def __init__(self, dim_num, feat_chan, prop_kernel, prop_step):
        self.dim_num = dim_num
        self.feat_chan = feat_chan
        self.prop_kernel = prop_kernel
        self.prop_step = prop_step

    def cspn(self, guide, feat):
        '''
            cspn func
        '''
        guide = fluid.layers.abs(guide)
        guide_num = guide.shape[1] // feat.shape[1]
        expand_times = [1, guide_num, *([1] * self.dim_num)]
        layer_name = 'cspn_affinity_propagate'
        if feat.shape[1] > 1:
            cspn_feat = list()
            for channel_ind in range(feat.shape[1]):
                slice_guide = fluid.layers.slice(
                    guide, axes=[1], starts=[channel_ind * guide_num],
                    ends=[(channel_ind + 1) * guide_num])
                normalizer = fluid.layers.reduce_sum(slice_guide, dim=1, keep_dim=True)
                normalizer = fluid.layers.expand(normalizer, expand_times=expand_times)
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
            normalizer = fluid.layers.expand(normalizer, expand_times=expand_times)
            guide = fluid.layers.elementwise_div(guide, normalizer)
            for _ in range(self.prop_step):
                feat = fluid.layers.affinity_propagate(
                    feat, gate_weight=guide, kernel_size=self.prop_kernel, name=layer_name)
            cspn_feat = feat
        return cspn_feat

    def demo(self, batch_size=3, iter_num=20, map_shape=None):
        '''
            func to run demo
        '''
        # define net
        if map_shape is None:
            map_shape = [48, 64, 128][3 - self.dim_num:]
        else:
            assert len(map_shape) == self.dim_num, '{}d map shape is required'.format(self.dim_num)
        guide_chan = self.prop_kernel ** self.dim_num - 1
        guide_shape = [self.feat_chan * guide_chan, *map_shape]
        guide = fluid.layers.data(name='guide', shape=guide_shape)
        feat_shape = [self.feat_chan, *map_shape]
        feat = fluid.layers.data(name='feat', shape=feat_shape, stop_gradient=False)
        cspn_feat = self.cspn(guide, feat)
        print(cspn_feat)
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
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--dimNum', choices=[2, 3], default=3, help='version of cspn module')
    PARSER.add_argument('--featChan', type=int, default=1, help='#channels of feature')
    PARSER.add_argument('--propKernel', choices=[3], default=3, help='kernel size for propagation')
    PARSER.add_argument('--propStep', type=int, default=24, help='#steps to propagate')
    ARGS = PARSER.parse_args()
    MODULE = CSPN(
        dim_num=ARGS.dimNum, feat_chan=ARGS.featChan,
        prop_kernel=ARGS.propKernel, prop_step=ARGS.propStep)
    MODULE.demo()

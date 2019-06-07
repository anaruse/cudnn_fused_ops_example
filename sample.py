import numpy

import cupy
from cupy.cuda import cudnn as libcudnn
from cupy import cudnn
from cupy import prof


n = 32

x_h = 28
x_w = 28
x_c = 64

k_h = 3
k_w = 3

y_h = 28
y_w = 28
y_c = 64

pad = (1, 1)
stride = (1, 1)

#

x = cupy.random.random((n, x_h, x_w, x_c)).astype(numpy.float16) - 0.5
x_desc = cudnn.create_tensor_descriptor(x, format=libcudnn.CUDNN_TENSOR_NHWC)

scale = cupy.random.random((1, x_c, 1, 1)).astype(numpy.float16)
bias = cupy.random.random((1, x_c, 1, 1)).astype(numpy.float16)
scale_desc = cudnn.create_tensor_descriptor(scale)

w = cupy.random.random((y_c, k_h, k_w, x_c)).astype(numpy.float16) - 0.5
filter_desc = cudnn.create_filter_descriptor(w,
                                             format=libcudnn.CUDNN_TENSOR_NHWC)

conv_desc = cudnn.create_convolution_descriptor(pad, stride, w.dtype,
                                                use_tensor_core=True)

y = cupy.empty((n, y_h, y_w, y_c)).astype(numpy.float16)
y_desc = cudnn.create_tensor_descriptor(y, format=libcudnn.CUDNN_TENSOR_NHWC)

ysum = cupy.empty((1, y_c, 1, 1)).astype(numpy.float32)
ysqsum = cupy.empty((1, y_c, 1, 1)).astype(numpy.float32)
ysum_desc = cudnn.create_tensor_descriptor(ysum)

act_desc = cudnn.create_activation_descriptor(libcudnn.CUDNN_ACTIVATION_RELU)
bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED
ops = libcudnn.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS

#

plan = cudnn.create_fused_ops_plan(ops)

const_pack = cudnn.create_fused_ops_const_param_pack(
    ops, ((libcudnn.CUDNN_PARAM_XDESC, x_desc),
          (libcudnn.CUDNN_PARAM_XDATA_PLACEHOLDER, ptr_ph),
          (libcudnn.CUDNN_PARAM_BN_MODE, bn_mode),
          (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, scale_desc),
          (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
          (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
          (libcudnn.CUDNN_PARAM_ACTIVATION_DESC, act_desc),
          (libcudnn.CUDNN_PARAM_CONV_DESC, conv_desc),
          (libcudnn.CUDNN_PARAM_WDESC, filter_desc),
          (libcudnn.CUDNN_PARAM_WDATA_PLACEHOLDER, ptr_ph),
          (libcudnn.CUDNN_PARAM_YDESC, y_desc),
          (libcudnn.CUDNN_PARAM_YDATA_PLACEHOLDER, ptr_ph),
          (libcudnn.CUDNN_PARAM_YSTATS_DESC, ysum_desc),
          (libcudnn.CUDNN_PARAM_YSUM_PLACEHOLDER, ptr_ph),
          (libcudnn.CUDNN_PARAM_YSQSUM_PLACEHOLDER, ptr_ph)))

workspace_size = cudnn.make_fused_ops_plan(plan, const_pack)
workspace = cupy.empty((workspace_size,), dtype=numpy.int8)
# print('workspace_size: {}'.format(workspace_size))

var_pack = cudnn.create_fused_ops_variant_param_pack(
    ops, ((libcudnn.CUDNN_PTR_XDATA, x),
          (libcudnn.CUDNN_PTR_BN_EQSCALE, scale),
          (libcudnn.CUDNN_PTR_BN_EQBIAS, bias),
          (libcudnn.CUDNN_PTR_WDATA, w),
          (libcudnn.CUDNN_PTR_YDATA, y),
          (libcudnn.CUDNN_PTR_YSUM, ysum),
          (libcudnn.CUDNN_PTR_YSQSUM, ysqsum),
          (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
          (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
           workspace_size)))

with prof.time_range('fusedOpsExecute', color_id=1, sync=True):
    cudnn.fused_ops_execute(plan, var_pack)

#

print('per-channel ysum:\n{}'.format(ysum.reshape((y_c))))
print('per-channel ysqsum:\n{}'.format(ysqsum.reshape(y_c)))

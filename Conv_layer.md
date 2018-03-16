# 卷积层

torch中的卷积操作有两种，分别为时域卷积和空间卷积：

- [时域卷机](https://github.com/torch/nn/blob/master/doc/convolution.md#temporalconvolution) ：对于一个输入序列，进行卷积操作。

- [空间卷积](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution)：对输入的2D或3D矩阵进行卷积。

在卷积神经网络中使用的为空间卷积，在THNN中对应[SpatialConvolutionMM.c函数](https://github.com/torch/nn/blob/master/lib/THNN/generic/SpatialConvolutionMM.c)。
卷积函数的输入参数如下：

```lua 
module = nn.SpatialConvolution(nInputPlane, nOutputPlane, 
                               kW, kH, [dW], [dH], [padW], [padH])
```

- 输入数据：nInputPlane，输入数据为3位矩阵，三维分别为通道数n_i，宽w_i，高h_i
- 输出数据：nOutputPlane，输入数据为3位矩阵，三维分别为通道数n_o，宽w_o，高h_o，其中输入与输出的通道数可能不等
- 卷积核参数：kW, kH，卷积核尺寸通常为正方形，但在某些网络中，可以使用1xN与Nx1的卷积核组合替代NxN的卷积核，从而减少运算量。
- 步长参数：dW, dH，卷积核在运算时的步长。
- 边缘填充参数：padW，padH，在卷积操作前在边缘填充0的数量。
其中，输入尺寸与输出尺寸的关系如下：
```lua
w_o  = floor((w_i  + 2*padW - kW) / dW + 1)
h_o = floor((h_i+ 2*padH - kH) / dH + 1)
```
在卷积操作运算前，通常要确定输出数据的尺寸，确保输出数据尺寸不为0，从而判断判断输入数据是否能够继续卷积

torch中对于卷积函数的实现方式为：

> nn(lua)->THNN(C)->THTensor(C)->THBlas(C)->LAPACK(Fortran)

目前主流框架计算卷积时，会采用BLAS（基础线性代数子程序库），将数据进行展开，使用其中的数学运算操作实现多维矩阵的计算。对于采用CUDA的系统，则会使用NVIDIA提供的cuBLAS库实现在CUDA平台的加速运算。

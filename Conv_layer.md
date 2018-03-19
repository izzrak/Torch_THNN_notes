# 卷积层

torch中的卷积操作有两种，分别为时域卷积和空间卷积：

- [时域卷机](https://github.com/torch/nn/blob/master/doc/convolution.md#temporalconvolution) ：对于一个输入序列，进行卷积操作。

- [空间卷积](https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialConvolution)：对输入的2D或3D矩阵进行卷积。

## 空间卷积
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

目前主流框架计算卷积时，会将数据进行展开，采用BLAS（基础线性代数子程序库），使用其中的数学运算操作实现多维矩阵的计算。对于采用CUDA的系统，则会使用NVIDIA提供的cuBLAS库实现在CUDA平台的加速运算。

### 计算流程
整个卷积的流程被封装在`THNN_(SpatialConvolutionMM_updateOutput_frame`这个函数里，分三步实现多维平面的卷积。分别为矩阵展开，偏移填充，矩阵相乘。
### 降维展开
由于输入数据通常是三维矩阵，而卷积则是对其中的平面进行操作，则可以将输入数据进行展开，生产一个较大的二位矩阵，从而实现加速计算。实现展开操作的函数是[unfold.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/unfold.c)中的`THNN_(unfolded_copy)`。
```cpp
THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH,
		     nInputPlane, inputWidth, inputHeight,
		     outputWidth, outputHeight);
```
其中，展开操作将输入input展开为finput，输出尺寸outputWidth, outputHeight可以根据卷积参数计算出来。finput的尺寸为`[n*kW*kH]*[outW*outH]`。

### 偏移填充
填充的目的是利用BLAS矩阵乘法的特点，先行构建输出矩阵，并将偏移值(bias)填充进去，实现卷积操作。
```cpp
// 定义一个[outN]*[outW*outW]的输出矩阵
output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                       nOutputPlane, -1,
                                       outputHeight*outputWidth, -1);
// 如果有bias不为0则填充bias数值，为零则填充为0。
if (bias) {
  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output->storage->data + output->storageOffset + output->stride[0] * i,
                    THTensor_(get1d)(bias, i), outputHeight*outputWidth);
} else {
  THTensor_(zero)(output);
}
```
### 矩阵相乘
将平移展开后的输入finput，通过与展开后的weight矩阵做乘法，得到卷积结果output，weight，finput，output矩阵关系如图所示：

![avatar](http://img.blog.csdn.net/20160831173531082)

矩阵相乘则使用[THTensorMath.c](https://github.com/torch/torch7/blob/master/lib/TH/generic/THTensorMath.c)中的`THTensor_(addmm)`函数实现
```cpp
THTensor_(addmm)(output2d, 1, output2d, 1, weight, finput);
```
该函数完成`output2d = 1*output2d + 1*weight*finput`的运算，其中output2d填充了bias，weight为复制展开的卷积核，finput为降维展开后的输入矩阵。
相乘步骤中的计算量为`[outN]*[outW*outW]`的偏移加法，`[outN]*[outW*outH]*[inN*kW*kH]`的乘法，`([inN*kW*kH]-1)*[outN]*[outW*outH]`的加法。

更深层次的计算，则依靠[THBlas_(gemm)](https://github.com/torch/torch7/blob/master/lib/TH/generic/THBlas.c)的实现。再经过一系列的尺寸，参数的合法性检查后，根据数据的浮点精度，判断传递给单精度矩阵计算`sgemm`，或者双精度矩阵计算`dgemm`。

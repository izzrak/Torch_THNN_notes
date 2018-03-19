# 简单非线性层
## 批归一化(BatchNormalization)
从字面意思看来Batch Normalization（简称BN）就是对每一批数据进行归一化
批归一化层只接受二维的输入。
```lua
              x - mean(x)
y =  ----------------------------- * gamma + beta
      standard-deviation(x) + eps
```

其中，每一个输入的通道N对应一个标准差，而gamma和beta为可学习的和通道数N 有关的数值。gamma和beta的学习性时可选参数。

在THNN中，使用`nn.BatchNormalization`调用函数，传递给[BatchNormalization.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/BatchNormalization.c)执行计算：
```lua
module = nn.BatchNormalization(N [, eps] [, momentum] [,affine])
```
其中 `N` 为输入的维度
`eps` 是加在标准差中的一个极小值，以防止除0的情况发生. 默认值为 `1e-5`.
`affine` 是一个boolean值. 当设为false时, 仿射变换的可学习性被关闭. 默认为true

批归一化主要的计算量在于计算平均值和方差，其中平均值由于训练数据读取的方式，很难直接求得，根据目前神经网络数据的送入方式(mini-batch)，可采用滑动平均。
所以，在训练过程中，做归一化时需要大量的计算，而在训练完成后，滑动平均值会保存在模型里，在应用模型时，只需要读取保存好的滑动平均，计算方差。

C语言核心代码如下：
### 计算平均值和标准差倒数
```cpp
// 计算平均值
accreal sum = 0;
TH_TENSOR_APPLY(real, in, sum += *in_data;);

mean = (real) sum / n;
THTensor_(set1d)(save_mean, f, (real) mean);
```
为了简化计算，直接求得标准差的倒数。
```cpp
// 计算标准差倒数
sum = 0;
TH_TENSOR_APPLY(real, in,
sum += (*in_data - mean) * (*in_data - mean););

if (sum == 0 && eps == 0.0) {
  invstd = 0;
} else {
  invstd = (real) (1 / sqrt(sum/n + eps));
}
THTensor_(set1d)(save_std, f, (real) invstd);
```
### 更新滑动平均与滑动方差
```cpp
// 更新滑动平均
THTensor_(set1d)(running_mean, f,
  (real) (momentum * mean + (1 - momentum) * THTensor_(get1d)(running_mean, f)));
//更新滑动方差
accreal unbiased_var = sum / (n - 1);
THTensor_(set1d)(running_var, f,
  (real) (momentum * unbiased_var + (1 - momentum) * THTensor_(get1d)(running_var, f)));
```
### 读取滑动的平均值与标准差倒数
```cpp
mean = THTensor_(get1d)(running_mean, f);
invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
```
### 利用向量乘法实现归一化
```cpp
// 计算归一化参数
real w = weight ? THTensor_(get1d)(weight, f) : 1;
real b = bias ? THTensor_(get1d)(bias, f) : 0;

TH_TENSOR_APPLY2(real, in, real, out,
*out_data = (real) (((*in_data - mean) * invstd) * w + b););
```
批归一化的计算量和当前卷积层的输出数据尺寸有关，batch大小为N，通道数为N，特征图大小为W和H，则做批归一化的数据量为`M*N*W*H`。其中计算标准差倒数时需要分别实现N个通道的标准差计算。而输入数据的每一个元素则需要进行两次加减法，两次乘法的计算。

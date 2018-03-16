# 传递函数(Transfer Function)
传递函数通常在参数转化层之后引入非线性关系，[将问题空间分割成更复杂的区域](https://github.com/torch/nn/blob/master/doc/transfer.md#transfer-function-layers)。在神经网络中则被称为激活函数(Activation Function)。

早期的激活函数(sigmoid, tanh)在浅层的神经网络中，具有很好的效果，但随着网络的复杂化，在卷积神经网络(CNN)中，ReLU(线性整流函数)具有结构简单，正向计算和反向传递运算量低等优点。

## ReLU

## SoftMax
[Softmax函数](https://en.wikipedia.org/wiki/Softmax_function)，或称归一化指数函数，是逻辑函数的一种推广。它能将一个含任意实数的K维的向量 
 _z_ 的“压缩”到另一个K维实向量 _σ(z)_ 中，使得每一个元素的范围都在(0, 1)之间，并且所有元素的和为1。

`Softmax` 定义如下:

```lua
f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
```
其中`shift = max_i(x_i)`，将指数计算固定在负数域，避免溢出。
在THNN中，使用nn.SoftMax调用函数，传递给[SoftMax.c](https://github.com/torch/nn/blob/master/lib/THNN/generic/SoftMax.c)执行计算
```lua
f = nn.SoftMax()
```
根据输入数据纬度的不同，softmax运算中涉及到的参数也不同

- 1-D：一维向量，在卷积神经网络中通常为
- 2-D：带有时序的一维向量，时序nframe，向量维度dim
- 3-D：三维矩阵，参数有维度，每个维度的长度和高度
- 4-D：带有时序的三位矩阵

在卷积神经网络中，Softmax常用于给予概率的多分类问题。

C语言的核心代码如下：
### 指针的定义
```cpp
// 使用指针直接对输入输出数据进行寻址
real *input_ptr = input_data + (t/stride)*dim*stride + t % stride;
real *output_ptr = output_data + (t/stride)*dim*stride + t % stride;
```
### 最大值查询
在对一维向量的操作中，需要对向量中的每个元素进行遍历，分别进行比较，比较的操作次数为dim的数值
```cpp
real inputMax = -THInf;
accreal sum;
// 遍历输入向量寻找最大值
 ptrdiff_t d;
for (d = 0; d < dim; d++)
{
  if (input_ptr[d*stride] >= inputMax) inputMax = input_ptr[d*stride];
}
```
### 完成Softmax函数的计算
计算指数时，需要先计算输入向量减去最大值的数值，在进行指数运算，之后对指数进行求和。在向量归一化中，需要对向量进行除法操作。则计算量为dim*(2*加减+1*指数+1*除法)
```cpp
sum = 0;
// 计算向量元素指数，并求和
for (d = 0; d < dim; d++)
{
  real z = exp(input_ptr[d*stride] - inputMax);
  output_ptr[d*stride] = z;
  sum += z;
}
// 向量归一化
for (d = 0; d < dim; d++)
{
  output_ptr[d*stride] *= 1/sum;
}
```

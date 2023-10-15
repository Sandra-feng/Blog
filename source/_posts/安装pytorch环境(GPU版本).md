# 安装pytorch环境(GPU版本)

离线安装比较简单

根据代码的需求进行配置环境。一般来说python=3.7版本的就已经可以了。

**第一步：查看电脑对应显卡cuda版本**

win+r打开运行，输入cmd打开命令行，输入nvidia-smi，查看cuda版本。

![image-20230927192716678](C:\Users\Hasee\AppData\Roaming\Typora\typora-user-images\image-20230927192716678.png)

说明我可以下载cuda12.0以下版本，cuda向下兼容。

CUDA版本与PyTorch版本之间的兼容性通常不是一一对应的，而是根据PyTorch的发布周期和支持策略来确定的。一般来说，PyTorch的不同版本会添加对不同CUDA版本的支持，但并不是每个PyTorch版本都与每个CUDA版本完全兼容。一般情况下，较新的CUDA版本通常可以兼容较旧的PyTorch版本。

例如，如果你安装了最新版本的CUDA，并且你选择一个较旧的PyTorch版本，通常情况下是可以正常工作的，因为较新的CUDA通常包含对较旧CUDA API的支持。然而，要注意的是，较旧的PyTorch版本可能不会利用较新CUDA版本的新特性和性能优化。

但是，反过来并不总是成立。较新的PyTorch版本可能会依赖于新的CUDA特性或API，因此可能不兼容较旧的CUDA版本。

总之，PyTorch和CUDA之间的兼容性通常是向后兼容的。

| **CUDA版本** | **PyTorch版本** |
| :----------- | :-------------- |
| 11.3         | 1.7.1           |
| 11.4         | 1.8.0           |
| 11.4         | 1.9.0           |
| 11.4         | 1.9.1           |
| 11.4         | 1.10.0          |

**第二步：在下面网站中找到对应的torch和torchvision，下载whl文件**

下载网站：https://download.pytorch.org/whl/torch_stable.html

在这里可以查看torch,torchversion的对应关系[mirrors / pytorch / vision · GitCode](https://gitcode.net/mirrors/pytorch/vision?utm_source=csdn_github_accelerator)

或者在这里：[以前的 PyTorch 版本 |PyTorch](https://pytorch.org/get-started/previous-versions/)

![image-20230927210637659](C:\Users\Hasee\AppData\Roaming\Typora\typora-user-images\image-20230927210637659.png)

这里出来的最新的版本对应关系是cuda11.8的，我们12.0的也完全够用。

由上图，我们来下载torch=2.0的，torchvision=0.15.0，torchaudio==2.0.0，我选择了python=3.9的，因为怕3.10以上的出问题。

![image-20230927210946774](C:\Users\Hasee\AppData\Roaming\Typora\typora-user-images\image-20230927210946774.png)

cu113表示cuda版本是11.3
cp37 表示python版本3.7
win-amd64 表示windows64位

下载完安装包后进行第三步

**第三步：打开anaconda prompt执行以下命令**

```python
 conda create -n torch2python39 python=3.9
```

`torch2python39`是自己命名的环境。

![image-20230927212647005](C:\Users\Hasee\AppData\Roaming\Typora\typora-user-images\image-20230927212647005.png)

```python
conda activate torch2python39
cd D:\Anaconda\安装包     #安装包文件夹
pip install torchvision-0.15.0+cu118-cp39-cp39-win_amd64.whl  #先安装了torchvision
pip install torchaudio-2.0.0+cu118-cp39-cp39-win_amd64.whl
pip install torchaudio-2.0.0+cu118-cp39-cp39-win_amd64.whl
```

最后激活python环境，输入`import torch`，`print(torch.cuda.is_available())`得到True说明torch环境安装成功。

![image-20230927214211715](C:\Users\Hasee\AppData\Roaming\Typora\typora-user-images\image-20230927214211715.png)
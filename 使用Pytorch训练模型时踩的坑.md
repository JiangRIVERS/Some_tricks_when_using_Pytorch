@author Jiang Rivers

# 使用Pytorch训练模型时踩的坑

### 1.指定gpu进行训练
当你有多个gpu可以进行训练时，你可以指定使用某一块卡进行训练或者指定某几张卡进行训练。
如果不指定的话，默认使用所有的卡进行训练。

在设置使用某块卡时，需要让其可见而其他卡不可见。

例如：
> 当你有编号为：0,1,2,3 共四张卡进行训练，你想使用1号gpu进行训练。这个时候输入
>
>      torch.cuda.device_count()
>则会返回4，代表现在可用4张卡，单卡时Pytorch默认使用0号gpu,多卡时默认使用4块卡。此时输入
>
>     os.environ["CUDA_VISIBLE_DEVICES"]="1"
>
>这行代码代表现在只有1号gpu是可见的，其余不可见
>
>此时
>
>     torch.cuda.device_count()
>则会返回1，Pytorch使用1号gpu进行训练。同理使用多卡。

这个地方有一个坑：

os.environ["CUDA_VISIBLE_DEVICES"]="1"这句应当在import torch之前，原因不明。
如果这句话在import后面，Pytorch依旧可见其余几块卡，单卡运行时依旧默认使用0号gpu，多卡时默认使用4块卡。

### 2.使用torch.utils.checkpoint.checkpoint减少占用显存

我在做一个项目的时候，使用12GB的gpu跑模型，但模型太大，会爆显存。于是使用了
checkpoint黑科技，成功将显存限制在4个多GB，模型终于可以跑啦。

checkpoint的原理是使用时间换空间，将模型分为两部分（或多个部分），分别跑两部分，将第一部分的结果给到第二部分开始，由
第二部分得到最终结果。

因为模型占用内存大多是无用的中间参数，这些占用的内存不会自动释放，
使用两阶段模型，可以只在第二部分输入第一部分的结果，而释放掉第一部分的无用参数。

### 3.使用Relu(inplace=True)减少内存消耗
如题

### 4.题外话 pip install 速度慢
换成国内镜像源
pip install -i +镜像源+package

镜像源：
>
> 清华：https://pypi.tuna.tsinghua.edu.cn/simple
>
> 阿里云：http://mirrors.aliyun.com/pypi/simple/
>
> 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
>
> 华中理工大学：http://pypi.hustunique.com/
>
> 山东理工大学：http://pypi.sdutlinux.org/ 
>
> 豆瓣：http://pypi.douban.com/simple/

例如：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyspider，这样就会从清华这边的镜像去安装pyspider库。
 




    
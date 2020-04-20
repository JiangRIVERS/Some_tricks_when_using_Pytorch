@author Jiang Rivers

# 使用Pytorch训练模型时踩的坑

### 1.指定gpu进行训练
当你有多个gpu可以进行训练时，你可以指定使用某一块卡进行训练或者指定某几张卡进行训练。
如果不指定的话，默认使用所有的卡进行训练。

在设置使用某块卡时，需要让其可见而其他卡不可见。

例如：
> 当你有编号为：0,1,2,3 共四张卡进行训练，你想使用1号gpu进行训练。这个时候输入
>
>```python
>torch.cuda.device_count()
>```
>则会返回4，代表现在可用4张卡，单卡时Pytorch默认使用0号gpu,多卡时默认使用4块卡。此时输入
>```python
>os.environ["CUDA_VISIBLE_DEVICES"]="1"
>```
>这行代码代表现在只有1号gpu是可见的，其余不可见
>
>此时
>```python
>torch.cuda.device_count()
>```
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

### 5.使用fastai的时候踩的坑
今天使用fastai，结果踩坑无数。
fastai可以通过一系列方式找到较为合适的学习率，详见论文

Cyclical Learning Rates for Training Neural Networks

fastai就是根据这篇文章的LR核心思想做出来的，可以说fastai和这篇论文属于互相成就，在
fastai之前，这篇论文并没有这么多关注量。

开始说坑：
一、我安装的是 1.0.61.dev0 版本，属于测试版。而官方的文档的稳定版的文档。测试版的文档将
有些类下的函数移到了不同的位置。比如说：在稳定版中basic_train类下有lr_find()函数，而测试版
则没有，但他还存在basic_train这个类，而且文档中并没有说明，就很迷。你使用了basic_train类，却告诉你没有lr_find()方法。

二、我之前通过checkpoint来牺牲时间换空间，使得模型能够在我的小12G的卡上跑，但是使用fastai却
不能使用这个功能。这是由于fastai太过于集成化，他致力于让神经网络不再复杂。
在pytorch下

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
model = models.resnet50(pretrained=True).to(device)
    
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)).to(device)
    
criterion = nn.CrossEntropyLoss()
    
optimizer = optim.Adam(model.fc.parameters())
    
learn = create_cnn(data, models.resnet50, metrics=accuracy)
```  
    
就可以了。这样看起来的确做到了让神经网络不再复杂的目的，但是这样就会导致一些问题。
比如checkpoint操作需要分两步计算网络输出，而不是直接由Pytorch调用
默认的forward()函数计算输出，而在fastai中，由于fastai过度集成化使得你无法进行类似的操作。

三、第三点其实和第二点很相似，我习惯于将网络输出之后再进行softmax()，然而在fastai中，
只能在搭建网络时将softmax加到foward()中，即fastai没有办法直接拿到output并进行操作。

总结下来，fastai可以作为一个寻找最优学习率的方法，但是往大了说还没有那么大的用处，毕竟
自适应optimizer已经很好用了，论文中说他们这种方式比自适应optimizer优点就是不需要增大计算量，
其实我觉得意义还没有那么大，只能说为optimizer选择初始学习率有些用处。

补充一句，这对新手还是很友好的，很集成，但是从现在的程度来看，只是新手友好以及
选择初始学习率较为有用。 

### 6. 训练时正常，测试时爆显存
经常遇到类似的情况，训练时很正常，12GB的GPU训练只占用了5GB，然而一到测试时就立刻爆显存，
具体情况没有了解特别清楚，大致理解为测试时的Tensor并不需要梯度，如果此时设置了梯度，模型
会加载训练时的大量梯度数据，导致爆显存。

解决方法:
在测试时
```python
with torch.no_grad():
    output=model(input)
```
加入 with torch.no_grad():将测试时的input梯度设为False，这样模型不会加载大量梯度信息。
从而解决了训练时正常，测试时爆显存的问题。

with torch.no_grad()和model.eval()的区别：
with torch.no_grad()是将Tensor的梯度设为False而model.eval()是用于在测试时
关闭BN层和Dropout层。

### 7. Pytorch 中retain_graph 参数的作用
参考：https://oldpan.me/archives/pytorch-retain_graph-work

retain_graph用于将计算图中的中间变量在计算完后保存，在平时使用中默认为False，
此时计算图中的中间变量在计算之后被释放，用于而提高效率。

但在特殊的场合，尤其是最近在研究GAN网络，这个时候有多个loss需要被bp算法
反向传播回去，这个时候，如果retain_graph是默认值False的话，参数会在
第一个loss反向传播之后被释放，导致后续的loss没办法反向传播，从而引发错误。
此时需要
```python
output1.backward(retain_graph=True)
```
而output2则不需要，即
```python
output2.backward()
```
即可。

### 8. Terminal 翻墙

在国内访问google等外网需要梯子，然而当我们使用shadowsocks之后，虽然网页可以正常
访问google了，但是terminal里还是不可以。导致在terminal里安装东西需要镜像，而在terminal
里访问google就更加的麻烦。

解决方法：在 ~/.bash_profile中加入
```
export http_proxy=socks5://127.0.0.1:1087
export https_proxy=socks5://127.0.0.1:1087
```
其中的127.0.0.1:1087需要和shadowsocks里的preference中相对应

或者更直接的将shadowsocks的全局模式打开

## 9. Pandas读取xlsx文件中多个sheet
当xlsx文件中有多个sheet时，如何读取某个sheet呢？
```python
xls = pd.ExcelFile('path_to_file.xls')
df1 = pd.read_excel(xls, 'Sheet1') # Sheet更换成你的sheet名字
df2 = pd.read_excel(xls, 'Sheet2')
```

## 10. unzip 中文文件出现乱码
使用公司电脑训练模型，公司电脑是windows,在压缩文件时默认编码是gbk，因而在
深度学习平台（Linux）进行unzip时出现乱码，当时想法特别蠢，写了段代码将文件名（中文）按
阿拉伯数字进行编码，在我沾沾自喜时，得知人家unzip直接有参数可以使用GBK编码读取...我....
```
unzip -O GBK xxx.zip
```

## 11. cv2.imread返回NoneType问题
问题一般是由文件名带中文，而windows下编码是gbk编码，所以导致中文不识别，问题不是cv2
不能读取路径，而是路径计算机就不识别。

解决方法
```python
import cv2
import numpy as np
cv2.imdecode(np.fromfile("D:\\中文名.png",dtype=np.uint8),cv2.IMREAD_COLOR)
```


当路径只有英文时，将'\'均改为'/'

例如：
```python
import cv2
cv2.imread("D:\\中文名.png")
```

改为：
```python
import cv2
cv2.imread("D:\\中文名.png")
```
同样保存图片时：
```python
im=cv2.imdecode(np.fromfile('c:\\测试\\1.jpg',dtype=np.uint8),cv2.IMREAD_UNCHANGED)#打开含有中文路径的图片
cv2.imencode('.jpg',im)[1].tofile('C:\\测试\\你好.jpg')#保存图片
```

所以尽量还是用英文编码，或者使用mac哈哈哈，毕竟读取文件时也涉及到'gbk'和'utf-8'的转换

## 12. 显示ValueError: Expected more than 1 value per channel when training, got input size [1, 512, 1, 1]
+ 当Batch_size不为1时，DataLoader如果数据的数量是奇数，batch_size无法整除，会出现该问题。例如：图片数量
为17张，然而batch_size为8，需要在DataLoader参数里面设置
```python
drop_last=True
```    
+ 当Batch_size为1时，可能是图片本身size太小，无法经过连续的卷积+下采样。

## 13. 关于DataLoader
在写自己的DataLoader子类的时候需要注意，如果是小数据集，可以将所有的数据都在__init__中
导入并进行预处理的操作，这个时候所有的数据都被读入内存中，运算相对更快。

但是当数据量大时，一定不要这么做，会OOM。正确的做法是在__init__函数中定义每个数据的url，
在__getitem__中进行读取和预处理。这样每次只IO一个数据并进行处理，不会出现OOM，但是每次涉及
IO操作，多少会慢一点。

## 14. CPU 爆memory
我在训练的时候有个习惯，将一个epoch的所有loss加在一起，得到的total_loss除以
每个epoch中所有的图片数量，得到这个epoch的平均loss，进行分析。

然而
```pyrhon
total_loss+=loss
``` 
这种操作会爆CPU的memory，因为每次计算得到的loss包含之前所有计算的Autograd，所以
随之循环次数的增加，total_loss会不断叠加到memory中，导致报错。

可以改为
```python
total_loss+=loss.item()
```
这样没吃total_loss仅仅加上一张图的loss,问题解决。


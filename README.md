# deep-learning-cheatsheet

虽然看过很多遍 cs-229, Coursera 上的 deep learning, 但是一直都没有好好地全部手写过一遍, 这个 repo 将放置各种奇奇怪怪的（~~从别人的 handbook 里搬来的~~）demo 和 tutorial 就当是 cheatsheet 了。

## TOC

* [chenyuntc/pytorch-book](https://github.com/chenyuntc/pytorch-book)
* [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
* [dsp-noise-analysis [PRIVATE]](https://github.com/JeffersonQin/c919-noise-analysis)
* [yolo-v1-pytorch](https://github.com/JeffersonQin/yolo-v1-pytorch)
* [yolo-v2-pytorch](https://github.com/JeffersonQin/yolo-v2-pytorch)
* tests - 一些测试
  * pytorch-data-augmentation - 测试 PyTorch 的图像增广

## Notes

* [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)
  * ch [3.7](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression-concise.html): [从精度问题看 softmax 与 CrossEntropyLoss](https://gyrojeff.top/index.php/archives/%E4%BB%8E%E7%B2%BE%E5%BA%A6%E9%97%AE%E9%A2%98%E7%9C%8B-softmax-%E4%B8%8E-CrossEntropyLoss/)
  * ch [4.5](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html), [4.6](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/dropout.html): [深度学习的正则化方法](https://gyrojeff.top/index.php/archives/deep-learning-regularization-methods/)
  * ch [4.7](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/backprop.html): [计算图的简单例子 - 参数复用](https://gyrojeff.top/index.php/archives/%E8%AE%A1%E7%AE%97%E5%9B%BE%E7%9A%84%E7%AE%80%E5%8D%95%E4%BE%8B%E5%AD%90-%E5%8F%82%E6%95%B0%E5%A4%8D%E7%94%A8/)
  * ch [4.9](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/environment.html): [深度学习分布偏移理论](https://gyrojeff.top/index.php/archives/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%88%86%E5%B8%83%E5%81%8F%E7%A7%BB%E7%90%86%E8%AE%BA/)
  * ch [5](https://zh-v2.d2l.ai/chapter_deep-learning-computation/index.html): [PyTorch 模块与参数 Cheatsheet](https://gyrojeff.top/index.php/archives/PyTorch-%E6%A8%A1%E5%9D%97%E4%B8%8E%E5%8F%82%E6%95%B0-Cheatsheet/)
* [YOLO v1](https://gyrojeff.top/index.php/archives/%E8%AF%BB%E8%AE%BA%E6%96%87YOLO-v1/)
* tests
  * [PyTorch Data Augmentation 数据增广](https://gyrojeff.top/index.php/archives/PyTorch-Data-Augmentation-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%B9%BF/)

## sync_scripts

这个文件夹下的可执行文件可以用于同步服务器中的文件。

* `sftp_commands.txt`: sftp batch file
* `sync_from_server.bat`: sync batch file
* `sync_from_server.sh`: sync bash script

须在 `sync_scripts` 目录下放置密钥 (`gpu.pem`) 用于验证。具体的同步命令请编辑 `sftp_commands.txt`。

若在脚本在 `wsl` 下运行出错，多半是 `chmod` 的问题。可以添加 (或新建文件) 如下两行到 `/etc/wsl.conf`:

```
[automount]
options = "metadata"
```

然后执行

```
wsl --shutdown
```

## Special Thanks

感谢 [@WetABQ](https://github.com/WetABQ) 提供的腾讯云 GPU 服务器。

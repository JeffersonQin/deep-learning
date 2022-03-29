# deep-learning-cheatsheet

虽然看过很多遍 cs-229, Coursera 上的 deep learning, 但是一直都没有好好地全部手写过一遍, 这个 repo 将放置各种奇奇怪怪的（~~从别人的 handbook 里搬来的~~）demo 和 tutorial 就当是 cheatsheet 了。

## TOC

* pytorch-book: https://github.com/chenyuntc/pytorch-book
* d2l-zh: https://github.com/d2l-ai/d2l-zh

## Notes

* d2l
  * ch 3.7: [从精度问题看 softmax 与 CrossEntropyLoss](https://gyrojeff.top/index.php/archives/%E4%BB%8E%E7%B2%BE%E5%BA%A6%E9%97%AE%E9%A2%98%E7%9C%8B-softmax-%E4%B8%8E-CrossEntropyLoss/)
  * ch 4.5, 4.6: [深度学习的正则化方法](https://gyrojeff.top/index.php/archives/deep-learning-regularization-methods/)

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

# CRNN-ATTENTION
crnn feature net + attention 机制

基于 tensorflow eager 格式编写。

数据集使用
mjsynth.tar.gz    http://www.robots.ox.ac.uk/~vgg/data/text/
需要将train_net.py 中 root 目录指向 mjsynth 中包含 annotation_train.txt 的文件夹

注意:tensorflow eager 保存好像是直接序列化层对象,导致GPU上训练的模型只能在GPU上使用,CPU同理

添加attention机制后,模型收敛很快,基本一个epoch就能基本收敛,所以可以考虑直接在CPU上训练测试

目前版本只能用于检测单个单词,对多个单词的情况基本预测全错



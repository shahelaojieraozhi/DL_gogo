from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log')

for i in range(100):
    writer.add_scalar("y=2x", 3 * i, i)  # add_scalar 添加标量
    # "y=2x"   tag (str): Data identifier     图的标题

writer.close()

'''
如何打开tensorboard： tensorboard --logdir=logs   (logs一定要换成绝对路径 这个路径不用带上引号，且该路径下必须包含事件文件)
'''

'''
换端口：tensorboard --logdir=logs --port=6007
'''


"""
参考：https://blog.csdn.net/qq_40128284/article/details/109343301
Q： 浏览器打开后没东西就显示：No dashboards are active for the current data set. 

1. 路径含有中文：
tensorboard --logdir=路径中含有中文可能会导致这个问题，解决方法也比较简单，将中文替换成英文。

2. 路径错误：
tensorboard --logdir=路径，这个路径不用带上引号，且该路径下必须包含事件文件。

3.启动tensorboard语句是否正确：
我使用的tensorboard版本为2.0.1，使用的语句为：tensorboard --logdir=E:\asa\20201028；E:\asa\20201028为事件文件的上一级目录，可以是绝对路径，也可以是相对路径，都不带引号。

有些版本的tensorboard需要将=改为“ ”，即：tensorboard --logdir ”E:\asa\20201028“，具体我没有去研究。
"""

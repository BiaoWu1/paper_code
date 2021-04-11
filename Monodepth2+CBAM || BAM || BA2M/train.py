# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
#absolute_import用来区分绝对引入和相对引入，调用import string表示使用系统的string；使用from pkg import string表示引入当前目录的string.py
#division， 精确除法 3/4 = 0.75 此时截断除法为  3//4 = 0
#print_function 限制输出的格式

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

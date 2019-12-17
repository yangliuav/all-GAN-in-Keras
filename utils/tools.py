import platform
import os
import numpy as np

def set_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax(memory_available) )

sysstr = platform.system()
if(sysstr =="Windows"):
    pass
else:
    set_freer_gpu()

categories = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']










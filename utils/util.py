import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def set_random_seed(seed="random"):

    if seed == "random":
        seed = random.randint(0, 9999)
    
    elif isinstance(seed, int):
        if seed <0:
            seed = random.randint(0, 9999)
        elif seed >9999:
            seed =seed % 9999

    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def plot_confusion_matrics (classes, confusion_matrix, save_path):
    confusion_matrix = np.array(confusion_matrix, dtype=int)
    count = len(classes)
    proportion=[]
    for i in confusion_matrix:
        for j in i:
            temp=j/(np.sum(i))
            proportion.append(temp)
    pshow=[]
    for i in proportion:
        pt="%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion=np.array(proportion).reshape(count,count)  # reshape(列的长度，行的长度)
    pshow=np.array(pshow).reshape(count,count)
    #print(pshow)
    plt.figure(dpi=100, figsize=(12,10))
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
                # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
                # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    plt.title('confusion_matrix', fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [item[:3] for item in classes], fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    thresh = confusion_matrix.max() / 2.
    #iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    #ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i,j] for j in range(count)] for i in range(count)],(confusion_matrix.size,2))
    for i, j in iters:
        if(i==j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12,color='black',weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12,color='black')
        else:
            plt.text(j, i-0.12, format(confusion_matrix[i, j]),va='center',ha='center',fontsize=12)   #显示对应的数字
            plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=12)
    
    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predict label',fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
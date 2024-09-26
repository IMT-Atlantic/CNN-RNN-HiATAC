from utils import *

import pandas as pd
import numpy as np
import torch.utils.data 
import torch
import pickle
import h5py as h5
import os
from model import *
from torch.autograd import Variable
import h5py 
import time
#wandb.init()

'''
！！！注意！！！这个脚本仅用于注释 因为我们在训练的过程中并没有声明使用高分辨数据集
输入数据：seq_length=TRAIN_SEQ_LENGTH, window_size=int(args.window_size), chroms=train_chroms, mode='train'
主程序前期输入：
LEARNING_RATE = float(args.lr)
    EXPERIMENT_VERSION = args.v
    LAMBDA = float(args.lam)
    TRAIN_SEQ_LENGTH = 200 
    TEST_SEQ_LENGTH = 200 
'''

# 三个部分：__init__; __len__; __getitem__
# 分别承担：数据的定义与初始化、数据总量统计与返回与数据处理的核心区

# 定义了一个继承自 torch.utils.data.Dataset 的自定义数据集类 Chip2HiCDataset
class Chip2HiCDataset(torch.utils.data.Dataset):
    # window_size为什么这么大？
    def __init__(self, chipseq_path=None, diag_list_dir=None, seq_length=200, window_size=14000, chroms=['chr22'], mode='train', save_dir='./Datasets', subtract_mean=False, obs_exp=False):
                
        # 初始化过程 设置额外的参数属性用于后期调用
        # 从save_path载入训练数据
        save_path_X = os.path.join(save_dir, 'H1_X.h5')
        save_path_y = os.path.join(save_dir, 'H1_y_5kb_Akita_ICED_microC.pickle')

        # 实例属性初始化：序列长度、染色体列表、200作为滑动长度以及窗口长度
        self.seq_length = seq_length
        self.chroms = chroms
        self.buf = 200
        self.window_size = window_size

        # 其他属性的初始化
        self.inputs = {}
        self.labels = {}
        self.sizes = []
        self.subtract_mean = subtract_mean
        self.obs_exp = obs_exp

        # 输入数据与标签
        # h5文件为输入数据 pickle文件为标签
        print("Loading input:")
        self.inputs = h5.File(save_path_X, 'r')
        print("Loading labels:")
        with open(save_path_y, 'rb') as handle:
            self.labels = pickle.load(handle)          

        # 计算数据段尺寸
        # 对每一条染色体计算对角线并打印长度【HiC数据集】
        # 根据轨迹数据和缓冲区大小确定每条染色体的数据段数目，追加到self.sizes
        for chr in self.chroms:
            diag_log_list = self.labels[chr]
            print(len(diag_log_list[0]))
            self.sizes.append((len(diag_log_list[0]) - 2*self.buf)//self.seq_length + 1)

        # 打印尺寸
        print(self.sizes)

        # 加载均值：如果 subtract_mean 或 obs_exp 为真，则加载均值用于后续标准化或对数变换
        if self.subtract_mean or self.obs_exp:
            self.mean = torch.load(os.path.join(save_dir, 'H1_5kb_Akita_ICE_microC_mean.pt')).numpy()

        return

    def __len__(self):
        # 计算并返回所有染色体分段数总和
        # len方法是pytorch的dataLoader函数一个核心方法 主要的作用是返回总样本数量 在迭代数据集中用于确定循环终止的条件
        return int(np.sum(self.sizes))


    # 根据索引index返回相应数据与标签
    # 处理数据集指定索引index的样本
    # 可用于实现数据的预处理和变换，例如数据增强、归一化等
    # 支持使用索引随机访问数据，这对于实现数据加载的多线程、多进程并行处理非常重要
    def __getitem__(self, index):
        
        # 计算数据段累积和 将所有不大于index的值设置为100000
        # 这是因为：Epigenomic signal tracks are first presented to the model in a sliding window fashion, with window size of 1.4 Mb and step size of 10 kb
        # np.argmin挖掘最小值索引 即是对应染色体
        arr = np.array(np.cumsum(self.sizes).tolist())
        arr[arr <= index] = 100000
        chrom_idx = np.argmin(arr)
        # 确定染色体
        chr = self.chroms[chrom_idx]
        # 计算当前段索引及其在缓冲区中的起止位置
        idx = int(index - ([0] + np.cumsum(self.sizes).tolist())[chrom_idx])
        start = idx*self.seq_length + self.buf
        end = np.minimum(idx*self.seq_length + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf)
        # 循环从起始到结束位置，调用 data_preparation 函数处理数据并追加至 contact_data 列表
        contact_data = []
        for t in range(idx*self.seq_length + self.buf, np.minimum(idx*self.seq_length + self.seq_length + self.buf, len(self.labels[chr][0]) - self.buf),1):
            contact_vec  = data_preparation(t,self.labels[chr],self.inputs[chr])
            contact_data.append(contact_vec)

        # 将 contact_data 转为 NumPy 数组
        # 使用 np.nan_to_num 替换缺失值为 0.0
        y_chr = np.array(contact_data)
        np.nan_to_num(y_chr, copy=False, nan=0.0)
        # 从输入数据中提取指定范围的片段，并转换为 float32 类型
        X_chr = self.inputs[chr][:5, 50*start-(self.window_size//2):50*end+(self.window_size//2)].astype('float32')

        # 从数据中减去均值 用于标准化
        if self.subtract_mean and y_chr.shape[0] > 0:
            y_chr = y_chr - self.mean

        # 对数变换：为什么？
        if self.obs_exp and y_chr.shape[0] > 0:
            y_chr = np.log(1 + (y_chr/self.mean))

        # 数据填充：如果标签数据长度小于 seq_length，则对其进行填充
        # 如果输入数据宽度小于预期长度 也对其进行填充
        if y_chr.shape[0] < self.seq_length:
  
            try:
                pad_y = np.zeros((self.seq_length - y_chr.shape[0], y_chr.shape[1]))
                y_chr = np.concatenate((y_chr, pad_y), axis=0)
            except:
                y_chr = np.zeros((self.seq_length,200))

            pad_X = np.zeros((X_chr.shape[0],self.seq_length*50+self.window_size - X_chr.shape[1]))
            X_chr = np.concatenate((X_chr, pad_X), axis=1)  

        # y_chr = y_chr * 10
        return X_chr.astype('float32'), y_chr.astype('float32')





# test_chroms = ['chr19', 'chr20', 'chr22']
# loader = Chip2HiCDataset(chroms=test_chroms)
# train_loader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=0)

# for x,y in train_loader:
#     print(x.shape)
#     print(y.shape)

"""
import time as time
import random 

# Initialize BERT
#net = DNABERT().cuda()
#head = FineTune().cuda()
#net.train(), head.train()
net = DNAEmbedding().cuda()

#print(net.state_dict()['dnabert.encoder.layer.11.output.dense.weight'])

"""
"""
# Create dataset
test_chroms = ["chr" + str(i) for i in range(1, 19)]
x = Chip2HiCDataset(chipseq_path='./ChipSeq_Data', diag_list_dir='./GM12878', seq_length=200, chroms=test_chroms, mode='train',prebuilt=True)
train_loader = torch.utils.data.DataLoader(x, batch_size=1, shuffle=False, num_workers=0)


#params = net.parameters() #list(net.parameters()) + list(head.parameters())
#optimizer = optim.Adam(params, lr=1e-3, weight_decay=0.0005)

# Generate one (Chip, DNA, Label) tuple
for i, (d,d1,l) in enumerate(train_loader):
    chip = d[0] 
    dna = d1[0]
    
    print(chip.shape)
    print(dna.shape)


t0 = time.time()
step = 20

"""
"""
out = net(dna.cuda())
out = torch.squeeze(out)
out = torch.cat((chip.cuda(), out), dim=0)
print(out.shape)

for ep in range(50):

    # Generate two random batches to be backpropagated through
    b1 = random.randint(0, 32000//30)
    b2 = random.randint(0, 32000//30)
    print(b1, b2)

    # Get embedding for sequence sequentially
    #dna_embedded = torch.Tensor(np.zeros((1,768,0)), requires_grad=True).cuda()  #requires_grad=True).cuda()
    dna_embedded = torch.empty(size = (1,768,0), requires_grad=True).cuda()
    for j in range(0,dna.shape[0],step):
        inp = dna[j:j+step].cuda()
        out = net(inp)
        if j == step*b1 or j == step*b2:
            dna_embedded = torch.cat((dna_embedded,out.detach()), dim=2)
        else:
            dna_embedded = torch.cat((dna_embedded,out.detach()), dim=2)
        
        #print(out.shape, dna_embedded.shape)
    

    print(net.state_dict()['dnabert.encoder.layer.11.output.dense.weight'])
    dna_embedded = head(dna_embedded)
    print(dna_embedded.shape)
    optimizer.zero_grad()
    loss = F.mse_loss(dna_embedded, chip.cuda())
    loss.backward()
    optimizer.step()
    wandb.log({'epoch': ep, 'loss': loss.item()})
    print("Loss: ", loss.item())

t1 = time.time()

print(t1 - t0)
#print(chip.shape, dna_embedded.shape, t1-t0)
    
"""





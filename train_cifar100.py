import json

with open('config.json', 'r') as file:
    config_info = json.load(file)

#DATA_init: not used for this code
FOLDER_init = config_info["FOLDER_init"]
#FOLDER_init = '/FOLDER_init/' #Location where this repogistory "ProxyDR" is located.

import argparse

parser = argparse.ArgumentParser(description= 'Train models for CIFAR-100 data')
parser.add_argument('--GPU'      , default=1, type=int,  help='GPU number')
parser.add_argument('--method'     , default='DR',        help='DR/normface/softmax')
parser.add_argument('--distance'     , default='euc',        help='euc/arccos')
parser.add_argument('--backbone'       , default='resnet50', help='backbone: inception / resnet50') 
parser.add_argument('--rand_backbone'       , action='store_true', help='randomly initialized backbone (instead of pretrained backbone)') 

parser.add_argument('--seed'      , default=1, type=int,  help='seed for training (, validation) and test data split')
parser.add_argument('--batch_size'      , default=32, type=int,  help='batch for training (, validation) and test')

parser.add_argument('--small'      , action='store_true',  help='use small training dataset (5000 training images)')
parser.add_argument('--use_val'      , action='store_true',  help='use validation set to find best model')
parser.add_argument('--aug'       , action='store_true', help='use augmentation') 
parser.add_argument('--resize'       , default=32, type=int, help='resize image size. Default=32 (no change). 224: Resnet50 default size. 299: Inception-v3 default size.') 
parser.add_argument('--clspri'   , action='store_true',  help='use class prior probability')
parser.add_argument('--ema'   , action='store_true',  help='use exponential moving average (EMA) for proxy')
parser.add_argument('--alpha'      , default=0.01, type=float,   help='alpha for EMA')
parser.add_argument('--mds_W'   , action='store_true',  help='use MDS (multi-dimensional scaling) for proxy')
parser.add_argument('--beta'      , default=1.0, type=float,   help='parameter beta for distance transformation T (only used in MDS). T(d)=pi*d/(beta+2*d) or T(d)=d/(beta+d)')
parser.add_argument('--dynamic'   , action='store_true',  help='update scale factor for AdaCos/AdaDR')
parser.add_argument('--CORR'   , action='store_true',  help='CORR loss proposed by Barz and Denzler (2019). It requires --mds_W==True to be turned on. Otherwise, it will be ignored. --dynamic==True has no effect')

params= parser.parse_args()

params.dataset='cifar100'



import torch
import numpy as np
import math
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os, time
os.environ["CUDA_VISIBLE_DEVICES"]=str(params.GPU)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

params.dataset='cifar100'
if params.dataset=='cifar100': 
    #DATA_PATH: not used in this code
    DATA_TAG = 'cifar'
else:
    print('Unknown dataset!')
    raise
    
if params.small:
    DATA_TAG = 'cifarS' #Use small training dataset

    
gray = False
    
data_cls = pd.read_csv(FOLDER_init+'ProxyDR/CIFAR100_cls.csv')
    

N_test =10000 #16.667%

if params.use_val:
    N_train = 45000 #75.000%
    N_val = 5000 #8.333%
    N= N_train+N_val+N_test #Number of images
else:
    N_train = 50000 #83.333%
    N= N_train+N_test #Number of images

if params.small:
    N_train = 5000
    if params.use_val:
        N_val= 45000 #25000
        N =  N_train+N_val+N_test #Number of images
    else:
        N=N_train+N_test #Number of images

n_classes = len(data_cls) #Number of clasees



data_cls2=data_cls.copy()
pd.options.mode.chained_assignment = None


max_len=1
for i in range(len(data_cls2)):
    tmp_Path = str(data_cls['Path'][i]) #.replace(DATA_PATH,'')
    data_cls2['Path'][i] = tmp_Path.split('/')
    if len(data_cls2['Path'][i])>max_len:
        max_len=len(data_cls2['Path'][i])

    
def calc_dist(l1, l2, mode='diff'):
    dist=0
    len1, len2 = len(l1), len(l2)
    len_min = np.minimum(len1, len2)
    chk_sm=-1
    for i in range(len_min):
        tmp_chk_sm=(l1[i]==l2[i])
        if tmp_chk_sm:
            chk_sm=i
        else:
            break
    if mode=='diff': #Only difference in hierarchy
        return (len1-chk_sm-1)+(len2-chk_sm-1)
    elif mode=='whole': #Assume each class has depth 10 in hierarchy
        return (max_len-chk_sm-1)+(max_len-chk_sm-1)    
    else:
        raise

tree_dist = np.ones([len(data_cls2),len(data_cls2)])*np.nan
for i in range(len(data_cls2)):
    for j in range(len(data_cls2)):
        tree_dist[i,j] = calc_dist(data_cls2['Path'][i], data_cls2['Path'][j], mode='diff')
        
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF


train_set_ = CIFAR100(root=FOLDER_init,train=True,
                                        download=False, transform=None) #transforms.ToTensor()

if params.use_val:
    train_set, val_set= torch.utils.data.random_split(train_set_, lengths=[N_train, N_val], generator=torch.Generator().manual_seed(params.seed)) 
else:
    train_set= train_set_
AUG=params.aug
if AUG:
    aug_eval=transforms.ToTensor()
    test_set = CIFAR100(root=FOLDER_init,train=False,
                                       download=False , transform=aug_eval)
else:
    test_set = CIFAR100(root=FOLDER_init,train=False,
                                       download=False , transform=transforms.ToTensor())



class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        img, label= self.dataset[index]
        img = self.map(img)
        return (img, label)

    def __len__(self):
        return len(self.dataset)

resize=params.resize
if resize!=32:
    train_set = MapDataset(train_set, transforms.Resize(resize))
    if params.use_val:
        val_set = MapDataset(val_set, transforms.Resize(resize))
    test_set = MapDataset(test_set, transforms.Resize(resize))

if AUG:
    aug = transforms.Compose([transforms.RandomAffine(degrees=15, 
                        translate=(0.1,0.1), 
                        scale=(0.9, 1.1), shear=10),
                        transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    train_set_tf = MapDataset(train_set, aug) 
else:
    train_set_tf = MapDataset(train_set, transforms.ToTensor()) 
if params.use_val:
    if AUG:
        val_set_tf = MapDataset(val_set,  aug_eval) 
    else:
        val_set_tf = MapDataset(val_set,  transforms.ToTensor()) 
test_set_tf = test_set 


trainloader = DataLoader(train_set_tf, batch_size=params.batch_size, shuffle=True)
if AUG:
    trainloader2 = DataLoader(MapDataset(train_set, aug_eval), batch_size=50, shuffle=True) #To analyze training
else:
    trainloader2 = DataLoader(MapDataset(train_set, transforms.ToTensor()), batch_size=50, shuffle=True) #To analyze training


if params.use_val:
    valloader = DataLoader(val_set_tf, batch_size=params.batch_size)
    if AUG:
        valloader2 = DataLoader(MapDataset(val_set, aug_eval), batch_size=200, shuffle=True) #To analyze training
    else:
        valloader2 = DataLoader(MapDataset(val_set, transforms.ToTensor()), batch_size=200, shuffle=True) 
testloader = DataLoader(test_set_tf, batch_size=params.batch_size)
testloader2 = DataLoader(test_set, batch_size=200, shuffle=True) #To analyze training


'''Analyze training set'''
use_cuda = torch.cuda.is_available()
labels_np=np.ones(0)*np.nan
for i, data in enumerate(trainloader):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
    labels_np=np.concatenate([labels_np, labels.cpu().numpy()],axis=0)
        
cls_unq, cls_cnt= np.unique(labels_np,return_counts=True)
log_cls_prior = np.log(cls_cnt/np.sum(cls_cnt))  #np.log(cls_cnt)-np.mean(np.log(cls_cnt))
inputs.shape, cls_unq, cls_cnt, len(cls_unq), log_cls_prior



'''Define backbones'''
from typing import Any
import torch.nn.functional as F
from torch import nn, Tensor

class Net(nn.Module):
    def __init__(self, distance='arccos'):
        super().__init__()
        self.W = F.normalize(torch.randn([n_classes,128])) 
        self.W = torch.nn.Parameter(self.W)
        self.distance= distance

    def forward(self):
        self.WN = self.W/(torch.norm(self.W, p=2, dim =1,keepdim=True)+ 1e-6)
        cos_sim_W = torch.mm(self.WN, self.WN.t()) 
        euc_W = ((2-2*cos_sim_W)+1e-12)**0.5
        arc_cos_W = torch.acos(cos_sim_W)
        if self.distance=='arccos':
            return arc_cos_W
        elif self.distance=='euc':
            return euc_W
        else:
            raise

if params.mds_W:
    net = Net(distance= params.distance)
    net = net.cuda()

    W_optim = torch.optim.Adam(net.parameters(),lr=1e-3, weight_decay=0.)
    if params.distance=='arccos':
        tree_dist2 = (np.pi/2)*tree_dist/(params.beta+tree_dist)
    elif params.distance=='euc':
        tree_dist2 = (2**0.5)*tree_dist/(params.beta+tree_dist)
    else:
        raise
    
    stresses_ = []
    for it in range(1000): 
        net.train()

        dist = torch.tensor(tree_dist2,dtype=torch.float)
        dist = dist.to('cuda')

        W_optim.zero_grad()

        dist_W = net()
        stress = torch.norm(dist-dist_W)/torch.norm(dist)
        #stress_sq = torch.norm(dist-arc_cos_W)**2/torch.norm(dist)**2
        stress.backward() 
        #stress_sq.backward()
        W_optim.step()

        stresses_.append(stress.detach().cpu().numpy())
                   
    net.eval()
    net.WN = F.normalize(net.W).detach()

###Need to edit/check from here (오후 6:20 2023-07-18)
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def Inception3(method):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=not params.rand_backbone)
    model.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        
    model.fc=torch.nn.Linear(in_features=2048, out_features=128, bias=True)  
    
    model.final_feat_dim = 128
    model.aux_logits=None
    model.AuxLogits=None
    model.transform_input=False
    if method in ['DR','normface']:
        model.W = F.normalize(torch.randn([n_classes,128])) 
        model.W = torch.nn.Parameter(model.W)
        #model._forward(x), _ = model._forward(x)
        model.scale = 10
    elif method=='softmax':
        model.W = torch.randn([n_classes,128])
        model.W = torch.nn.Parameter(model.W)
        model.b = torch.zeros([1, n_classes]) #bias term
        model.b = torch.nn.Parameter(model.b)
    else:
        raise
    return model

def ResNet50(method):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=not params.rand_backbone)
    #model = torchvision.models.resnet50(pretrained=True, progress=True) 
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    #model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False) 
    model.fc = nn.Linear(in_features=512 * 4, out_features=128, bias=True) #in_features=512 * block.expansion
        
    #model.final_feat_dim = 128
    #model.aux_logits=None
    #model.AuxLogits=None
    #model.transform_input=False
    if method in ['DR','normface']:
        model.W = F.normalize(torch.randn([n_classes,128])) 
        model.W = torch.nn.Parameter(model.W)
        #model._forward(x), _ = model._forward(x)
        model.scale = 10
    elif method=='softmax':
        model.W = torch.randn([n_classes,128])
        model.W = torch.nn.Parameter(model.W)
        model.b = torch.zeros([1, n_classes]) #bias term
        model.b = torch.nn.Parameter(model.b)
    else:
        raise
    return model



def update_proxy(model_, feats_n, y_, alpha=0.01):
    with torch.no_grad():
        '''Prepare x_outN2'''
        
        W1=model_.W.clone()
        y2, inv_ind, counts=torch.unique(y_, sorted=True, return_inverse=True, return_counts=True)
        arg_y = torch.argsort(y_)
        if torch.max(counts)>1: #when there are multiple points with the same class
            feats_n2 = torch.ones(size=[len(counts),feats_n.size(1)],device=feats_n.device)

            mk_l=[] #marker list (same length with y)
            dup_l=[] #duplicate list (based on the length of y)
            i2=0 
            for i in range(len(counts)):
                if counts[i]==1:
                    mk_l.append(True)
                    i2+=1
                else:
                    for j in range(counts[i]):
                        mk_l.append(False)
                    dup_l.append(i2)
                    i2+=int(counts[i])
            
            mk_l=torch.Tensor(mk_l)
            mk_l = mk_l.type(torch.bool)
            
            dup_l2= torch.nonzero(counts>1)[:,0] #duplicate list (based on the length of counts)
            feats_n2[counts==1]=feats_n[arg_y][mk_l] #when there is only one point per class

            dup_counts=counts[counts>1]

            for i in range(len(dup_counts)):
                feat_tmp=feats_n[arg_y][dup_l[i]:dup_l[i]+dup_counts[i]].mean(dim=0) #Use average to update proxy when there are multiple points with the same class
                feats_n2[dup_l2[i]] = F.normalize(feat_tmp,dim=0) #feat_tmp/(torch.norm(feat_tmp, p=2)+ 0.00001) 
        else:
            feats_n2 = feats_n[arg_y]            

        tmp_EMA_marker=model_.EMA_marker[y2]#1 or 0 
        tmp_EMA_marker=tmp_EMA_marker.unsqueeze(dim=1)  
        tmp_EMA_marker=tmp_EMA_marker.cuda()

        pre_newpos_y = (alpha*feats_n2+(1-alpha)*model_.W[y2]) #new position (using EMA)

        pre_W_y = tmp_EMA_marker*feats_n2 +(1.0-tmp_EMA_marker)*pre_newpos_y #(self.alpha*x_outN2+(1-self.alpha)*self.classifier.customW[y2])
        pre_W_yN = F.normalize(pre_W_y) #pre_W_y/(torch.norm(pre_W_y, p=2, dim =1,keepdim=True)+ 0.00001)
        model_.W[y2]=pre_W_yN
        model_.EMA_marker[y2]=False
        W2=model_.W.clone()

        
def analyze_feat(model, feats_n, y):
    #model.W: [n_classes, dim] ,feats_n: [batch_size, dim]
    WN = F.normalize(model.W).detach()
    cos_sim_feat_W = torch.mm(feats_n.detach(), WN.t()) 
    arccos_feat_W_ = torch.acos(0.999999*torch.clip(cos_sim_feat_W,-1,1)) #[batch_size, n_classes]
    arccos_feat_W = arccos_feat_W_[torch.arange(len(feats_n),dtype=int),y] #[batch_size]
    
    marker = torch.ones_like(arccos_feat_W_).detach() #[batch_size, n_classes]
    marker[torch.arange(len(feats_n),dtype=int), y] = np.nan #0.0
    arccos_feat_W_df = arccos_feat_W_*marker #arccos for different classes
    
    arg_sort = torch.argsort(arccos_feat_W_,dim=-1) #[batch_size, n_classes]
    d_NN, _ = torch.min(arccos_feat_W_,dim=-1) #torch.acos(x_arccos_[arg_sort[]]) #Using nearest neighbor
    P_1 = WN[arg_sort[:,0],:] #[batch_size, dim] Nearest proxy
    s = torch.ones(feats_n.size(0))*np.inf #[batch_size]
    s = s.to(WN.device)
    zero_s = torch.zeros_like(s)
    zero_s = zero_s.to(WN.device)
    for i in range(1,WN.size(0)):
        P_i = WN[arg_sort[:,i],:] #[batch_size, dim]
        tmp_s = (1-torch.sum(P_1*P_i,dim=-1))/(1-torch.sum(P_1*P_i,dim=-1) +torch.sum(feats_n.detach()*(P_i-P_1),dim=-1))
        cond=(tmp_s>0)
        s = torch.where(cond, torch.min(s, tmp_s), torch.max(zero_s,s))
    s_ = torch.unsqueeze(s, dim=1)
    B_ = P_1 + s_*(feats_n.detach()-P_1)
    B = F.normalize(B_) #B/(tmp_norm+ 1e-6) #[N, dim]
    d_B = torch.acos(torch.sum(feats_n.detach()*B,dim=-1)) #Distance to the classification boundary
    I=d_B/(d_NN+d_B) #[batch_size] Intensity: it is not described in the paper. High intensity (close to 1) means a feature is relatively close to the nearest proxy. Low intensity (close to 0) means a feature is relatively close to the classification boundary. 
    
    
    cos_sim_W_W = torch.mm(WN, WN.t())
    euc_W  = (2-2*cos_sim_W_W+1e-12)**0.5
    arccos_W_W = torch.acos(0.999999*torch.clip(cos_sim_W_W,-1,1)) #[n_classes, n_classes]
    arccos_W_W[torch.arange(len(WN),dtype=int), torch.arange(len(WN),dtype=int)] = np.nan #torch.nan
    
    arccos_feat_W, I, arccos_W_W = arccos_feat_W.cpu().numpy(), I.cpu().numpy(), arccos_W_W.cpu().numpy()
    arccos_feat_W_df = arccos_feat_W_df.cpu().numpy()
    return arccos_feat_W, arccos_feat_W_df, I, arccos_W_W, euc_W.cpu().numpy()
    
def get_discriminality(model, loader):
    global use_cuda 
    #Related papers "Negative Margin Matters: Understanding Margin in Few-shot Classification" (www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490426.pdf)
    # and "Revisiting Training Strategies and Generalization Performance in Deep Metric Learning" (proceedings.mlr.press/v119/roth20a/roth20a.pdf)
    
    feats_np = np.ones([0,128])*np.nan
    labels_np = np.ones([0])*np.nan
    model.eval()
    for i, data in enumerate(loader):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        feats = model(inputs.detach()) #torch.Size([32, 128])        
                
        feats_n = F.normalize(feats).detach()
        feats_np = np.concatenate([feats_np, feats_n.cpu().numpy()],axis=0)
        labels_np = np.concatenate([labels_np, labels.cpu().numpy()],axis=0)
        
    indexes = [np.where(labels_np==i)[0] for i in range(len(data_cls))]
    proto_ = [np.mean(feats_np[indexes[i]],axis=0,keepdims=True) for i in range(len(data_cls))]
    mu = np.concatenate(proto_, axis=0) #[48, 128] #Before normalization
    proto_ = [proto_[i]/(np.linalg.norm(proto_[i],axis=-1,keepdims=True)+1e-12) for i in range(len(data_cls))]
    proto = np.concatenate(proto_, axis=0) #[48, 128]
    np.random.seed(params.seed)
    for i in range(len(data_cls)):
        if len(indexes[i])==0:
            feats_rand = np.random.normal(size=128)
            proto[i,:] = feats_rand/(np.linalg.norm(feats_rand,keepdims=True)+1e-12)
    
    model.proto = torch.tensor(proto,dtype=torch.float,device = model.W.device)

    WN = F.normalize(model.W.detach())
    W_np = WN.cpu().numpy()
    
    mu_dot = np.matmul(mu, mu.transpose()) #[48,48]
    mu_sq = np.sum(mu**2,axis=1,keepdims=True) #[48,1]
    if np.sum(np.isnan(mu_sq))>=1:
        nonnan_ind = np.logical_not(np.isnan(mu_sq)[:,0])
        mu_dot_ = mu_dot[nonnan_ind,:] #[?, 48]
        sum_inter = 2*mu_sq[nonnan_ind]-2*mu_dot_[:,nonnan_ind] #[?, ?]
        len_mu = np.sum(nonnan_ind)
    else:
        sum_inter = 2*mu_sq-2*mu_dot #[48, 48]
        len_mu = len(mu) #np.sum(nonnan_ind)
    
    var_inter = np.sum(sum_inter)/(len_mu*(len_mu-1))
    
    feat_sqs = [np.sum(feats_np[indexes[i]]**2,axis=1,keepdims=True) for i in range(len(data_cls))] #Each element: [N_i,1]
    feat_dots = [np.matmul(feats_np[indexes[i]], mu[i, None].transpose()) for i in range(len(data_cls))] #Each element: [N_i,1]
    sum_intra = [np.sum(feat_sqs[i]+mu_sq[i,None]-2*feat_dots[i])/len(indexes[i]) for i in range(len(data_cls)) if len(indexes[i])>=1] #Each element: [N_i,1]->1
    var_intra = np.sum(sum_intra)/len_mu #len(mu)    
    
    cos_sim_ = np.matmul(proto, W_np.transpose())
    arccos_ = np.arccos(np.clip(cos_sim_,-1,1))
    arccos_proto_W = arccos_[np.arange(len(proto)),np.arange(len(proto))]
    return var_inter, var_intra, arccos_proto_W




chk_depth=0
def higher_cls(inds, level=0, return_list=False, verbose=0):
    global chk_depth
    h_cls_list=[]
    for i in range(len(data_cls2)):
        depth = len(data_cls2['Path'][i])
        if depth>level:
            h_cls_list.append(data_cls2['Path'][i][level])
        else:
            if chk_depth==0:
                print('Some label will be used instead of higher level class (due to \'depth<=level\')!\n   depth:',depth, 'level:',level)
            h_cls_list.append(data_cls2['Path'][i][-1])
            chk_depth=1
    #print(set(h_cls_list))
    unq_inds=np.unique(h_cls_list, return_index=True)[1]
    h_cls_list_unq = [h_cls_list[ind_] for ind_ in sorted(unq_inds)]
    #h_cls_list_unq = sorted(set(h_cls_list), key=h_cls_list.index) #Without repeats
    h_cls_list_unq = np.array(h_cls_list_unq)
    h_cls_list = np.array(h_cls_list)
    cls2ind = {cls: ind for ind, cls in enumerate(h_cls_list_unq)}
    
    if verbose==1:
        print('Number of higher classes:',len(h_cls_list_unq))
    elif verbose==2:
        print('Number of higher classes:',len(h_cls_list_unq), '\ncls2ind:',cls2ind) #h_cls_list_unq)
    else:
        pass
    h_cls = h_cls_list[inds]
    
    h_cls_ind= list(map(lambda x: cls2ind[x], h_cls))
    if return_list:
        return h_cls, h_cls_ind, h_cls_list_unq 
    else:        
        return h_cls, h_cls_ind 
    
    
def get_acc_measures(W,level=0,distance='euc',each=True):
    #Measures explained in the paper "A Metric Learning Reality Check" (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf)
    _,h_cls_ind,_=higher_cls(np.arange(len(W)),level,True,verbose=0)
    h_cls_ind = np.array(h_cls_ind)
    
    cos_sim = np.matmul(W, W.transpose())
    arccos = np.arccos(np.clip(cos_sim,-1,1))
    euc = (2*(1-np.clip(cos_sim,-1,1)))**0.5 #Euclidean distance
    if distance=='euc':
        dist=euc
    elif distance=='arccos':
        dist=arccos
    else:
        raise
    indexes = [np.where(h_cls_ind==i)[0] for i in np.unique(h_cls_ind)]
    
    dist_rank = np.argsort(dist)[:,1:] 
    if each:
        P_1s=[]
        RPs = []
        MAP_Rs =[] 
    cnt=3*[0]
    cnt_tot=3*[0]
    for i in np.unique(h_cls_ind):
        if len(indexes[i])>1:
            for j in indexes[i]:
                tmp_cnt=3*[0]
                if dist_rank[j, 0] in indexes[i]: #The nearest class
                    tmp_cnt[0]=1
                    cnt[0]+=1
                intersect=np.intersect1d(indexes[i],dist_rank[j, :len(indexes[i])-1]) #R=len(indexes[i])-1
                
                tmp_Ps=[]
                for k in range(1,len(indexes[i])):
                    tmp_intersect=np.intersect1d(indexes[i],dist_rank[j, :k])
                    if dist_rank[j, k-1] in indexes[i]:
                        tmp_Ps.append(len(tmp_intersect)/k)
                    else:
                        tmp_Ps.append(0)
                cnt[1]+=len(intersect)
                if each:
                    P_1s.append(tmp_cnt[0])
                    RPs.append(len(intersect)/(len(indexes[i])-1))
                    MAP_Rs.append(np.mean(tmp_Ps))
            cnt_tot[0]+=len(indexes[i])
            cnt_tot[1]+=len(indexes[i])*(len(indexes[i])-1)
            
            
    if each:
        P_1 = np.mean(P_1s)
        RP = np.mean(RPs)
        MAP_R = np.mean(MAP_Rs)
    else:
        raise
        P_1 = cnt[0]/cnt_tot[0]
        RP = cnt[1]/cnt_tot[1]
        MAP_R = cnt[2]/cnt_tot[2]
    return P_1, RP, MAP_R



def analyze_scale_change(feats1, feats2):
    feats1_n = F.normalize(feats1) #[batch_size, dim]
    feats2_n = F.normalize(feats2) #[batch_size, dim]

    alpha_ = torch.trace(torch.matmul(feats1_n.t(),feats2_n))/self.torch_norm(feats1_n)**2
    norm_ratio=self.torch_norm(feats2_n-alpha_*feats1_n)/self.torch_norm(feats2_n-feats1_n)
    return alpha_, norm_ratio

def analyze_feat_change(W1, feats1, W2, feats2, y_):    
    WN1 = F.normalize(W1) #[n_classes, dim]
    WN2 = F.normalize(W2)
    
    feats1_n = F.normalize(feats1) #[batch_size, dim]
    feats2_n = F.normalize(feats2) #[batch_size, dim]
    
    cos_sim1 = torch.mm(feats1_n, WN1.t()) #[batch_size, n_classes]
    cos_sim2 = torch.mm(feats2_n, WN2.t())
    
    if params.distance=='arccos':
        arccos1 = torch.acos(0.999999*torch.clip(cos_sim1,-1,1))
        arccos2 = torch.acos(0.999999*torch.clip(cos_sim2,-1,1))
        dist1 = arccos1 #[batch_size, n_classes]
        dist2 = arccos2
    elif params.distance=='euc':
        euc_sq1 = 2-2*cos_sim1
        euc_sq2 = 2-2*cos_sim2
        dist1 = (euc_sq1+1e-12)**0.5 #[batch_size, n_classes]
        dist2 = (euc_sq2+1e-12)**0.5
    else:
        raise

    dist_ratio = dist2/dist1 #[batch_size, n_classes]
    log_ratio_sm=[torch.log(dist_ratio[i,y_[i]]) for i in range(len(feats1))] #log ratio for same (correct) classes
    log_ratio_diff=[torch.log(dist_ratio[i,j]) for i in range(len(feats1)) for j in range(len(W1)) if y_[i]!=j] #log ratio for different (incorrect) classes
    converg = torch.exp(torch.mean(torch.stack(log_ratio_sm, dim=0))).cpu().numpy() #Geometric mean of rate of convegence
    diverg = torch.exp(torch.mean(torch.stack(log_ratio_diff, dim=0))).cpu().numpy() #Geometric mean of rate of divergence
    con_div_ratio = converg/diverg
    return converg, diverg, con_div_ratio



'''Train model'''
method = params.method 
dist = params.distance 
backbone = params.backbone 
rand_backbone=params.rand_backbone
seed = params.seed
batch_size= params.batch_size
small = params.small
use_val = params.use_val
clspri = params.clspri #Whether to use class prior probability or not
ema = params.ema 
if ema:
    alpha=params.alpha 
mds_W = params.mds_W
if mds_W:
    beta = params.beta
dynamic = params.dynamic 
CORR = params.CORR

if backbone == 'inception':
    model = Inception3(method=method)
elif backbone == 'resnet50':
    model = ResNet50(method=method)
else:
    raise

if clspri:
    model.log_cls_prior = torch.tensor(log_cls_prior)
    model.log_cls_prior = torch.unsqueeze(model.log_cls_prior,dim=0) #[1, n_classes]
    
if ema:
    model.W.requires_grad=False
    #Use EMA_marker to use feature positions as initial W (initialize W using feature positions)
    model.EMA_marker = torch.ones(len(model.W)) 
    model.EMA_marker.requires_grad=False
    
if mds_W:
    model.W = torch.nn.Parameter(net.WN.cpu())
    model.W.requires_grad=False
    
if dynamic:
    from scipy.optimize import fsolve
    if method == 'DR':
        if dist=='euc':
            sc_func = lambda x : ((n_classes-1.0)/2**(x/2))*(x+1)*(2-2**0.5)**(x/2)-x+1
        elif dist=='arccos':
            sc_func = lambda x : ((n_classes-1.0)/(np.pi/2)**x)*(x+1)*(np.pi/4)**x-x+1
        else:
            raise
    elif method == 'normface':
        if dist=='euc':
            sc_func = lambda x : np.cos(np.pi/4)*(n_classes-1+np.exp(x*np.cos(np.pi/4)))+x*np.sin(np.pi/4)**2*(np.exp(x*np.cos(np.pi/4))-n_classes+1)
        elif dist=='arccos':
            sc_func = lambda x : (n_classes-1)*np.exp(-x*(np.pi/2)**2)*np.exp(2*x*(np.pi/4)**2)*(2*x*(np.pi/4)**2-1)-np.exp(x*(np.pi/4)**2)*(2*x*(np.pi/4)**2+1)       
        else:
            raise
    else:
        raise
    model.scale = fsolve(sc_func, model.scale)[0]
    model.log_scale = torch.nn.Parameter(torch.tensor([np.log(model.scale)],dtype=torch.float)) 
    model.log_scale.requires_grad = True

model = model.cuda()
if clspri:
    model.log_cls_prior = model.log_cls_prior.to(model.W.device)

if backbone == 'inception':
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=0.) #lr=1e-4: Inception3
    if dynamic:
        optimizer0 = torch.optim.Adam([model.log_scale],lr=1e-4, weight_decay=0.) 
elif backbone == 'resnet50':
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=0.)
    if dynamic:
        optimizer0 = torch.optim.Adam([model.log_scale],lr=1e-4, weight_decay=0.) 
else:
    raise
    
criterion = nn.CrossEntropyLoss() 
use_cuda = torch.cuda.is_available()
losses_ = []

model_PATH_ = 'models_'+DATA_TAG+'/'
if use_val:
    model_PATH_ += backbone+'_'+method+'_vsd'+str(seed)
else:
    model_PATH_ += backbone+'_'+method+'_sd'+str(seed)

if rand_backbone:
    model_PATH_ = model_PATH_.replace(backbone,'rd'+backbone)
    
if (method=='DR')&(dist=='euc'):
    model_PATH_ = model_PATH_.replace(method, method+'_euc')
if (method=='normface')&(dist=='arccos'):
    model_PATH_ = model_PATH_.replace(method, method+'_arccos') 

if clspri:
    model_PATH_ += '_clspri'
if ema:
    model_PATH_ += '_ema'
if mds_W:
    model_PATH_ += '_mds'
if dynamic:
    model_PATH_ += '_dnm'
if CORR:
    model_PATH_ += '_CORR'
if AUG:
    model_PATH_ += '_aug'
if resize!=32:
    model_PATH_ += '_rsz'+str(resize)
old_stdout = sys.stdout
print(model_PATH_)
log_file = open(model_PATH_.replace('models_'+DATA_TAG+'/','record/'+DATA_TAG+'_')+'.txt','w')
sys.stdout = log_file

print('Method:',method)
if method in ['DR','normface']:
    print('Distance:',dist)
print('Backbone:',backbone)
print('Seed:',params.seed)
print('Small CIFAR100:',small)
print('Use validation:',use_val)
print('Augmentation:',AUG)
if method in ['DR','normface']:
    print('Consider class prior:',clspri)
    print('EMA:',ema)
    if ema:
        print('   alpha:',alpha)
    print('MDS_W:',mds_W)
    if mds_W:
        print('   beta:',beta)
        print('   Stress (normalized):',stresses_[0],stresses_[-1])
        if ema:
            print('   (EMA is ignored when \'mds_W==True\'.)')
        print('   CORR:',CORR)
    print('Dynamic scale:',dynamic)
    if dynamic:
        print('Scale (initial):',model.scale)
    else:
        print('Scale:',model.scale)
#print(model_PATH_)
print()
sys.stdout.flush()


from scipy.stats import spearmanr, combine_pvalues, hmean

def fisher_corr(r):
    r_np = np.array(r)
    z=np.arctanh(r_np)
    z_mn = np.nanmean(z)
    return np.tanh(z_mn)

def corr_from_DMs(dm1, dm2, classwise=True):
    if classwise:
        corr_list=[]
        p_value_list=[]
        
        for i in range(len(dm1)):
            tmp_corr = spearmanr(dm1[:,i], dm2[:,i]) 
            corr_list.append(tmp_corr[0])
            p_value_list.append(tmp_corr[1])
        
        p_value_np =np.array(p_value_list)
        nn_ind=np.logical_not(np.isnan(p_value_np))
        p_value_np = p_value_np[nn_ind]
        return fisher_corr(corr_list), hmean(p_value_np) #combine_pvalues(p_value_list)[-1] #,method='stouffer'
    else: #Elementwise correlation
        dm1_vec=np.ones([0])*np.nan
        dm2_vec=np.ones([0])*np.nan
        
        for i in range(len(dm1)):
            for j in range(i+1,len(dm2)): 
                dm1_vec = np.concatenate([dm1_vec, np.array([dm1[i,j]])])
                dm2_vec = np.concatenate([dm2_vec, np.array([dm2[i,j]])])
        
        return spearmanr(dm1_vec, dm2_vec)   

acc_np=np.zeros([0,3,2])*np.nan
corr_np = np.zeros([0,2])*np.nan
acc_measures_np=np.zeros([0,4,3,2])*np.nan
best_acc=0
t1 = time.time()

if small:
    tot_epoch=100 #1000
else:
    tot_epoch=100

for epoch in range(tot_epoch):  # loop over the dataset multiple times    
    corrects=0
    cnt=0
    corrects_v=0
    corrects_v_proto=0
    cnt_v=0
    corrects2=0
    corrects2_proto=0
    cnt2=0
    loc_t1 = time.time()
    loc_analyze=np.zeros([0,3])
    model.train()
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        
        #sys.stdout.flush()
        # forward + backward + optimize
        feats = model(inputs) #torch.Size([32, 128])
        
        if (i%(len(trainloader)//10)==0)&(method in ['DR','normface']):
            W1= model.W.detach()
            feats1 = feats.detach()
        
        if method in ['DR','normface']:
            feats_n = feats/torch.norm(feats, p=2, dim =1,keepdim=True) #F.normalize(feats)
            if ema:
                cos_sim = torch.mm(feats_n, F.normalize(model.W.detach()).t())
            else:
                cos_sim = torch.mm(feats_n, (model.W/torch.norm(model.W, p=2, dim =1,keepdim=True)).t())
                
            if dist=='arccos':
                arccos = torch.acos(0.999999*torch.clip(cos_sim,-1,1))
                if method=='DR':
                    pre_score = -torch.log(arccos+1e-12)
                else:
                    pre_score = -arccos**2
            else: #dist=='euc'
                euc_sq = 2-2*cos_sim
                if method=='DR':
                    pre_score = -torch.log(euc_sq+1e-12)/2
                else:
                    pre_score = cos_sim #-euc_sq
                    
                if dynamic:
                    arccos = torch.acos(0.999999*torch.clip(cos_sim,-1,1))               
                    
            if dynamic:
                score = torch.exp(model.log_scale.detach())*pre_score
            else:
                score = model.scale*pre_score
        elif method=='softmax':
            score = torch.mm(feats, model.W.t())+model.b #[32, 48]: [batch_size, n_classes]
        else:
            raise
            
        if clspri:
            score +=model.log_cls_prior
        
        if mds_W&CORR:
            cos_sim_corr = torch.diag(cos_sim[:,labels])     
            loss = torch.sum(1-cos_sim_corr) #criterion(score, labels)
        else:   
            loss = criterion(score, labels)

        loss.backward()
        optimizer.step()
        
        if ema&(mds_W==False):
            update_proxy(model, feats_n, labels, alpha=alpha)
            
        if dynamic:
            #This part is modified from https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
            optimizer0.zero_grad() 
            
            theta = arccos.detach() #[batch_size, n_classes]
            one_hot = torch.ones_like(arccos,dtype=bool)
            one_hot[torch.arange(len(labels),dtype=int),labels] = False
            
            if dist=='arccos':
                if method=='DR':
                    numer = 1/theta**torch.exp(model.log_scale.detach()) #Numerator 
                elif method=='normface':
                    numer = torch.exp(-torch.exp(model.log_scale.detach())*(arccos.detach())**2)
                else:
                    raise
            else: #dist=='euc'
                if method=='DR':
                    numer = 1/(euc_sq.detach())**torch.exp(model.log_scale.detach()/2)
                elif method=='normface':
                    numer = torch.exp(torch.exp(model.log_scale.detach())*cos_sim.detach())
                else:
                    raise
            if clspri:
                numer *= torch.exp(model.log_cls_prior)
            B_avg = torch.where(one_hot, numer.detach(), torch.zeros_like(arccos))
            B_avg = torch.sum(B_avg)/labels.size(0)
            theta_med = torch.median(theta[torch.arange(len(labels),dtype=int),labels])
            pi_torch = torch.tensor(np.pi/4)
            pi_torch = pi_torch.to(theta_med.device)
            theta_med = torch.min(pi_torch, theta_med)
            
            if clspri:
                sm_pri = torch.exp(model.log_cls_prior[labels])[0,:] #[batch_size]
            else:
                sm_pri = torch.ones_like(labels, dtype=torch.float) #[batch_size]
            avg_sm_pri = torch.mean(sm_pri.detach())
                    
            #raise
            
            euc_sq_theta_med = 2-2*torch.cos(theta_med.detach())
            
            if dist=='arccos':
                if method=='DR':
                    loss_sc= (B_avg.detach()*(torch.exp(model.log_scale)+1)*theta_med.detach()**torch.exp(model.log_scale)+avg_sm_pri*(1-torch.exp(model.log_scale)))**2
                elif method=='normface':
                    loss_pt1 = B_avg.detach()*torch.exp(torch.exp(model.log_scale)*euc_sq_theta_med)*(2*torch.exp(model.log_scale)*euc_sq_theta_med-1)
                    loss_pt2 = -avg_sm_pri*(2*torch.exp(model.log_scale)*euc_sq_theta_med+1)
                    loss_sc= (loss_pt1+loss_pt2)**2
                else:
                    raise
            else: #dist=='euc'
                if method=='DR':
                    loss_sc = (B_avg.detach()*(torch.exp(model.log_scale)+1)*(euc_sq_theta_med+1e-12)**(torch.exp(model.log_scale)/2)+avg_sm_pri*(1-torch.exp(model.log_scale)))**2
                elif method=='normface':
                    loss_pt1 = avg_sm_pri*torch.exp(torch.exp(model.log_scale)*torch.cos(theta_med.detach()))*(torch.cos(theta_med.detach())+torch.exp(model.log_scale)*torch.sin(theta_med.detach())**2)
                    loss_pt2 = B_avg.detach()*(torch.cos(theta_med.detach())-torch.exp(model.log_scale)*torch.sin(theta_med.detach())**2)
                    loss_sc= (loss_pt1+loss_pt2)**2
                else:
                    raise
            loss_sc.backward()
            optimizer0.step()            

        losses_.append(loss.detach().cpu().numpy())
        
        corrects+=torch.sum(torch.argmax(score,dim=-1)==labels)
        cnt+=len(labels)
        
        if (i%(len(trainloader)//10)==0)&(method in ['DR','normface']):
            model.eval()

            W2= model.W.detach()

            feats = model(inputs) 

            feats2 = feats.detach()
            
            tmp_analyze = analyze_feat_change(W1, feats1, W2, feats2, labels.detach()) 
            tmp_analyze = np.array([tmp_analyze])
            loc_analyze = np.concatenate([loc_analyze, tmp_analyze],axis=0)

            model.train()
    
    #Evaluation on training data    
    model.eval()
    print('Epoch:',epoch)
    t2= time.time()
    print('   Spend time (total):',t2-t1)
    print('   Spend time (local training):',t2-loc_t1)
    if dynamic:
        print('   Scale:',torch.exp(model.log_scale).detach().cpu().numpy()[0])
        print('      B_avg:',B_avg.detach().cpu().numpy())
    print('   Loss:', losses_[-1])
    
    inputs, labels = next(iter(trainloader2))
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    feats = model(inputs) 
    
    feats_n = feats/torch.norm(feats, p=2, dim =1,keepdim=True) #F.normalize(feats)
    
    arccos_feat_W, arccos_feat_W_df, I, arccos_W_W, euc_W = analyze_feat(model, feats_n, labels)
    
    print('      Intensity (mean): {0},  Intensity (std): {1}'.format(np.nanmean(I), np.nanstd(I)))
    
    sys.stdout.flush()
    if method in ['DR','normface']:
        var_inter, var_intra, arccos_proto_W = get_discriminality(model, trainloader)
        print('      var_inter: {0},  var_intra: {1}'.format(var_inter,var_intra))
        print('      discriminality:',var_inter/var_intra)
        sys.stdout.flush()
    
    print('      arccos (feats, W); same: {0}, diff: {1}'.format(np.mean(arccos_feat_W), np.nanmean(arccos_feat_W_df)))
    print('      arccos (W, W); min: {0}, mean: {1}'.format(np.mean(np.nanmin(arccos_W_W,axis=1)), np.nansum(arccos_W_W)/(len(arccos_W_W)*(len(arccos_W_W)-1))))
    sys.stdout.flush()
    if method in ['DR','normface']:
        print('      arccos (proto, W):',np.mean(arccos_proto_W))
        sys.stdout.flush()
        
    acc_train=(corrects/cnt).detach().cpu().numpy()
    print('   Accuracy (train):',acc_train)
    sys.stdout.flush()
    
    
    if method in ['DR','normface']:
        print('      converg:',loc_analyze[:,0])
        print('      diverg:',loc_analyze[:,1])
        print('      con_div_ratio:',loc_analyze[:,2])
        
        cos_sim_proto = torch.mm(model.proto, model.proto.t())
        euc_proto = (2-2*cos_sim_proto+1e-12)**0.5
        euc_proto = euc_proto.cpu().numpy()
        corr_info1 = corr_from_DMs(euc_W, tree_dist, classwise=True)
        print('      Mean corr (W):', corr_info1)
        corr_info2 = corr_from_DMs(euc_proto, tree_dist, classwise=True)
        print('      Mean corr (proto):', corr_info2)
        sys.stdout.flush()
        tmp_corr = np.array([[corr_info1[0], corr_info2[0]]])
        corr_np = np.concatenate([corr_np, tmp_corr],axis=0)
        
        tmp_acc_measures_np = np.zeros([4,3,2])*np.nan
        for i in range(0,4):
            P_1, RP, MAP_R=get_acc_measures(model.W.detach().cpu().numpy(),level=i,distance=params.distance,each=True)
            #print('        n_clusters:',i,'W',P_1, RP, MAP_R)
            P_1_, RP_, MAP_R_=get_acc_measures(model.proto.detach().cpu().numpy(),level=i,distance=params.distance,each=True)
            #print('        n_clusters:',i,'proto',P_1_, RP_, MAP_R_)
            tmp_acc_measures_np[i] = np.array([[P_1, RP, MAP_R],[P_1_, RP_, MAP_R_]]).transpose() #,axis=0) 
        acc_measures_np=np.concatenate([acc_measures_np,np.expand_dims(tmp_acc_measures_np,axis=0)],axis=0)          
        
    sys.stdout.flush()
    
    #Evaluation on validation data    
    model.eval()
    if use_val:
        for i, data in enumerate(valloader):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            feats = model(inputs) #torch.Size([32, 128])

            if method in ['DR','normface']:
                feats_n = feats/torch.norm(feats, p=2, dim =1,keepdim=True) #F.normalize(feats)
                '''Based on model.W'''
                if ema:
                    cos_sim = torch.mm(feats_n, F.normalize(model.W.detach()).t())
                else:
                    cos_sim = torch.mm(feats_n, (model.W/torch.norm(model.W, p=2, dim =1,keepdim=True)).t())
                if dist=='arccos':
                    arccos = torch.acos(0.999999*torch.clip(cos_sim,-1,1))
                    if method=='DR':
                        pre_score = -torch.log(arccos+1e-12)
                    else:
                        pre_score = -arccos**2
                else: #dist=='euc'
                    euc_sq = 2-2*cos_sim
                    if method=='DR':
                        pre_score = -torch.log(euc_sq+1e-12)/2
                    else:
                        pre_score = cos_sim #-euc_sq

                    if dynamic:
                        arccos = torch.acos(0.999999*torch.clip(cos_sim,-1,1))    

                if dynamic:
                    score = torch.exp(model.log_scale.detach())*pre_score
                else:
                    score = model.scale*pre_score

                '''Based on model.proto'''
                if ema:
                    cos_sim_proto = torch.mm(feats_n, F.normalize(model.proto).t())
                else:
                    cos_sim_proto = torch.mm(feats_n, (model.proto/torch.norm(model.proto, p=2, dim =1,keepdim=True)).t())
                if dist=='arccos':
                    arccos_proto = torch.acos(0.999999*torch.clip(cos_sim_proto,-1,1))
                    if method=='DR':
                        pre_score_proto = -torch.log(arccos_proto+1e-12)
                    else:
                        pre_score_proto = -arccos_proto**2
                else: #dist=='euc'
                    euc_sq_proto = 2-2*cos_sim_proto
                    if method=='DR':
                        pre_score_proto = -torch.log(euc_sq_proto+1e-12)/2
                    else:
                        pre_score_proto = cos_sim_proto #-euc_sq  

                if dynamic:
                    score_proto = torch.exp(model.log_scale.detach())*pre_score_proto
                else:
                    score_proto = model.scale*pre_score_proto

            elif method=='softmax':
                score = torch.mm(feats, model.W.t())+model.b #[batch_size, n_classes]
            else:
                raise        

            if clspri:
                score +=model.log_cls_prior
                score_proto +=model.log_cls_prior
            corrects_v+=torch.sum(torch.argmax(score,dim=-1)==labels)
            if method in ['DR','normface']:
                corrects_v_proto+=torch.sum(torch.argmax(score_proto,dim=-1)==labels)
            cnt_v+=len(labels)
        acc_val = (corrects_v/cnt_v).detach().cpu().numpy()            
        print('   Accuracy (val, W):',acc_val)
        if method in ['DR','normface']:
            acc_val_proto=(corrects_v_proto/cnt_v).detach().cpu().numpy()
            print('   Accuracy (val, proto):',acc_val_proto)
        
        if acc_val>best_acc:     
            print('   Save best model!') 
            best_acc = acc_val
            model_best_PATH = model_PATH_+'_best.pth'
            torch.save(model.state_dict(), model_best_PATH)
            
        
        inputs, labels = next(iter(valloader2))
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        feats = model(inputs) #torch.Size([32, 128])

        feats_n = feats/torch.norm(feats, p=2, dim =1,keepdim=True)#F.normalize(feats)

        arccos_feat_W, arccos_feat_W_df, I, arccos_W_W, _ = analyze_feat(model, feats_n, labels)
        print('      Intensity (mean): {0},  Intensity (std): {1}'.format(np.nanmean(I), np.nanstd(I)))
        print('      arccos (feats, W); same: {0}, diff: {1}'.format(np.mean(arccos_feat_W), np.nanmean(arccos_feat_W_df)))
        
        
        
    #Evaluation on test data
    for i, data in enumerate(testloader):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        feats = model(inputs) #torch.Size([32, 128])
        
        if method in ['DR','normface']:
            feats_n = feats/torch.norm(feats, p=2, dim =1,keepdim=True) #F.normalize(feats)
            '''Based on model.W'''
            if ema:
                cos_sim = torch.mm(feats_n, F.normalize(model.W.detach()).t())
            else:
                cos_sim = torch.mm(feats_n, (model.W/torch.norm(model.W, p=2, dim =1,keepdim=True)).t())
            if dist=='arccos':
                arccos = torch.acos(0.999999*torch.clip(cos_sim,-1,1))
                if method=='DR':
                    pre_score = -torch.log(arccos+1e-12)
                else:
                    pre_score = -arccos**2
            else: #dist=='euc'
                euc_sq = 2-2*cos_sim
                if method=='DR':
                    pre_score = -torch.log(euc_sq+1e-12)/2
                else:
                    pre_score = cos_sim #-euc_sq
                    
                if dynamic:
                    arccos = torch.acos(0.999999*torch.clip(cos_sim,-1,1))    
                
            if dynamic:
                score = torch.exp(model.log_scale.detach())*pre_score
            else:
                score = model.scale*pre_score
                
            '''Based on model.proto'''
            if ema:
                cos_sim_proto = torch.mm(feats_n, F.normalize(model.proto).t())
            else:
                cos_sim_proto = torch.mm(feats_n, (model.proto/torch.norm(model.proto, p=2, dim =1,keepdim=True)).t())
            if dist=='arccos':
                arccos_proto = torch.acos(0.999999*torch.clip(cos_sim_proto,-1,1))
                if method=='DR':
                    pre_score_proto = -torch.log(arccos_proto+1e-12)
                else:
                    pre_score_proto = -arccos_proto**2
            else: #dist=='euc'
                euc_sq_proto = 2-2*cos_sim_proto
                if method=='DR':
                    pre_score_proto = -torch.log(euc_sq_proto+1e-12)/2
                else:
                    pre_score_proto = cos_sim_proto #-euc_sq  
                
            if dynamic:
                score_proto = torch.exp(model.log_scale.detach())*pre_score_proto
            else:
                score_proto = model.scale*pre_score_proto
                
        elif method=='softmax':
            score = torch.mm(feats, model.W.t())+model.b #[batch_size, n_classes]
        else:
            raise        
            
        if clspri:
            score +=model.log_cls_prior
            score_proto +=model.log_cls_prior
        corrects2+=torch.sum(torch.argmax(score,dim=-1)==labels)
        if method in ['DR','normface']:
            corrects2_proto+=torch.sum(torch.argmax(score_proto,dim=-1)==labels)
        cnt2+=len(labels)
    acc_test=(corrects2/cnt2).detach().cpu().numpy()
    print('   Accuracy (test, W):',acc_test)
    if method in ['DR','normface']:
        acc_test_proto = (corrects2_proto/cnt2).detach().cpu().numpy()
        print('   Accuracy (test, proto):',acc_test_proto)
    
    
    inputs, labels = next(iter(testloader2))
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    feats = model(inputs) #torch.Size([32, 128])
    
    feats_n = feats/torch.norm(feats, p=2, dim =1,keepdim=True)#F.normalize(feats)
    
    arccos_feat_W, arccos_feat_W_df, I, arccos_W_W, _ = analyze_feat(model, feats_n, labels)
    print('      Intensity (mean): {0},  Intensity (std): {1}'.format(np.nanmean(I), np.nanstd(I)))
    print('      arccos (feats, W); same: {0}, diff: {1}'.format(np.mean(arccos_feat_W), np.nanmean(arccos_feat_W_df)))
    
    if method in ['DR','normface']:
        tmp_acc = np.array([[[(corrects/cnt).detach().cpu().numpy(), np.nan],[acc_val, acc_val_proto],[acc_test, acc_test_proto]]])
        
    else:
        tmp_acc = np.array([[[(corrects/cnt).detach().cpu().numpy(), np.nan],[acc_val, np.nan],[acc_test, np.nan]]])
        
    acc_np = np.concatenate([acc_np, tmp_acc],axis=0)
    np.savez(model_PATH_+'.npz', acc_np=acc_np, corr_np=corr_np, acc_measures_np=acc_measures_np)
    #print()

    plt.figure(1)
    plt.semilogy(losses_)
    plt.savefig('loss_plot_cifar',dpi=200)
    plt.close()
    print()
    sys.stdout.flush()
    
sys.stdout = old_stdout
log_file.close()
        

model_PATH = model_PATH_+'.pth'
torch.save(model.state_dict(), model_PATH)

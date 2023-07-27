import json

with open('config.json', 'r') as file:
    config_info = json.load(file)

#DATA_init: not used for this code
FOLDER_init = config_info["FOLDER_init"]
#FOLDER_init = '/FOLDER_init/' #Location where this repogistory "Inspecting_Hierarchies_ML" is located.

import argparse

parser = argparse.ArgumentParser(description= 'Evaluate models for NABirds data')
parser.add_argument('--GPU'      , default=2, type=int,  help='GPU number')
parser.add_argument('--method'     , default='DR',        help='DR/normface/softmax')
parser.add_argument('--distance'     , default='euc',        help='euc/arccos')
parser.add_argument('--backbone'       , default='resnet50', help='backbone: inception / resnet50') 
parser.add_argument('--rand_backbone'       , action='store_true', help='randomly initialized backbone (instead of pretrained backbone)') 

parser.add_argument('--seed'      , default=1, type=int,  help='seed for training (, validation) and test data split')
#parser.add_argument('--batch_size'      , default=32, type=int,  help='batch for training (, validation) and test')
parser.add_argument('--small'      , action='store_true',  help='use small training dataset (5000 training images)')
parser.add_argument('--use_val'      , action='store_true',  help='use validation set to find best model')
parser.add_argument('--last'      , action='store_true',  help='use the last (epoch) model for evaluattion')

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


params.dataset='nabirds'
params.batch_size=32
#print(params.method, params.backbone, params.ema)



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
#from sklearn.metrics import top_k_accuracy_score #Requires python version 3.7 or newer

params.dataset='nabirds'
if params.dataset=='nabirds': 
    DATA_TAG = 'nabird'
else:
    print('Unknown dataset!')
    raise
    
if params.small:
    DATA_TAG = 'nabirdS'
    
gray = False
    
    
data_info = pd.read_csv(FOLDER_init+'Inspecting_Hierarchies_ML/'+params.dataset+'_info.csv')
data_cls = pd.read_csv(FOLDER_init+'Inspecting_Hierarchies_ML/nabirds_cls2.csv',delimiter='|')
 
N= len(data_info) #Number of images: 48562

if params.use_val:
    N_train = int(0.7*N) #70%
    N_val = int(0.1*N) #10%
    N_test = N-N_train-N_val #20%
else:
    N_train = int(0.9*N) #90%
    N_test = N-N_train #10%

n_classes = len(data_cls) #Number of clasees

if params.small:
    N_train = 5000
    if params.use_val:
        N_val= 30000 
        N_test=N-N_train-N_val #13562
    else:
        N_test=N-N_train

n_classes = len(data_cls) #Number of clasees



data_cls2=data_cls.copy()
pd.options.mode.chained_assignment = None

max_len=1
for i in range(len(data_cls2)):
    tmp_Path = data_cls2['Hierarchy'][i] #.replace(DATA_PATH,'')
    data_cls2['Hierarchy'][i] = tmp_Path.split('//')
    if len(data_cls2['Hierarchy'][i])>max_len:
        max_len=len(data_cls2['Hierarchy'][i])
    
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
        tree_dist[i,j] = calc_dist(data_cls2['Hierarchy'][i], data_cls2['Hierarchy'][j], mode='diff')       
        
        
        
chk_depth=0
def higher_cls(inds, level=0, return_list=False, verbose=0):
    global chk_depth
    h_cls_list=[]
    for i in range(len(data_cls2)):
        depth = len(data_cls2['Hierarchy'][i])
        #print(depth)
        if depth>level:
            h_cls_list.append(data_cls2['Hierarchy'][i][level])
        else:
            if chk_depth==0:
                print('Some label will be used instead of higher level class (due to \'depth<=level\')!\n   depth:',depth, 'level:',level)
            h_cls_list.append(data_cls2['Hierarchy'][i][-1])
            chk_depth=1
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



from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

class NabirdsDataset(Dataset):
    def __init__(self, info_path, cls_path, transform=None):
        self.data_info = pd.read_csv(info_path)
        self.data_cls = pd.read_csv(cls_path)
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.data_info.iloc[index,0]
        img = Image.open(img_path).convert('RGB')
        cls = self.data_info.iloc[index,1]
        label = torch.tensor(int(self.data_cls.loc[self.data_cls['Class']==cls,'Number']))
        if self.transform is not None:
            img = self.transform(img)    
            
        if params.resize!=32:
            img = TF.resize(img, size=[params.resize,params.resize]) #size=[299,299]) #
        return (img, label)

    def __len__(self):
        return len(self.data_info)

nabirds_dataset = NabirdsDataset(info_path=FOLDER_init+'Inspecting_Hierarchies_ML/'+params.dataset+'_info.csv', 
                              cls_path=FOLDER_init+'Inspecting_Hierarchies_ML/'+params.dataset+'_cls.csv',transform=None)
if params.use_val:
    train_set, val_set, test_set = torch.utils.data.random_split(nabirds_dataset, lengths=[N_train, N_val, N_test], generator=torch.Generator().manual_seed(params.seed)) 
else:
    train_set, test_set = torch.utils.data.random_split(nabirds_dataset, lengths=[N_train, N_test], generator=torch.Generator().manual_seed(params.seed)) 




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


AUG=params.aug
if AUG:
    #In eval, we don't use augmentation
    aug_eval=transforms.ToTensor()
    train_set_tf = MapDataset(train_set, aug_eval)
else:
    train_set_tf = MapDataset(train_set, transforms.ToTensor())
if params.use_val:
    val_set_tf = MapDataset(val_set,  transforms.ToTensor())
test_set_tf = MapDataset(test_set,  transforms.ToTensor())    
    
    
    

    


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
testloader2 = DataLoader(test_set, batch_size=200, shuffle=True)

        
        
'''Analyze training set'''
use_cuda = torch.cuda.is_available()
labels_np=np.ones(0)*np.nan
t1=time.time()
if params.clspri:
    for i, data in enumerate(trainloader2):
        # get the inputs; data is a list of [inputs, labels]
        if i%100==0:
            t2=time.time()
            print(i)
            print('Spend time:', t2-t1)
        inputs, labels = data

        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        labels_np=np.concatenate([labels_np, labels.cpu().numpy()],axis=0)

    cls_unq, cls_cnt= np.unique(labels_np,return_counts=True)
    log_cls_prior = np.log(cls_cnt/np.sum(cls_cnt))  #np.log(cls_cnt)-np.mean(np.log(cls_cnt))


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
        euc_W = (torch.clip(2-2*cos_sim_W,0,4)+1e-12)**0.5
        arc_cos_W = torch.acos(torch.clip(cos_sim_W,-1,1)*(1-1e-6)) #torch.acos(torch.clip(cos_sim_W,-1,1)) 
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
        
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
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





        
    
    
def get_discriminality(model, loader):
    global use_cuda 
    
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
    
    model.proto = torch.tensor(proto,dtype=torch.float,device = model.W.device) #Without NaNs

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
    var_intra = np.sum(sum_intra)/len_mu
    
    
    cos_sim_ = np.matmul(proto, W_np.transpose())
    arccos_ = np.arccos(np.clip(cos_sim_,-1,1))
    arccos_proto_W = arccos_[np.arange(len(proto)),np.arange(len(proto))]
    return var_inter, var_intra, arccos_proto_W

def top_k_correct(label, score, k=5):
    #label: [n], score: [n,n_classes]
    y_preds=np.argsort(-score,axis=-1)[:,:k] #[n,k]
    correct_np=np.zeros_like(label,dtype=bool)
    for i in range(k):
        correct_np=np.logical_or(correct_np,y_preds[:,i]==label)
        
    return np.sum(correct_np)
    
def top_k_AHD(label, score, k=5):  
    #label: [n], score: [n,n_classes]
    #global tree_dist
    y_preds=np.argsort(-score,axis=-1)[:,:k] #[n,k]
    AHD_sum=0.0
    for i in range(k):
        AHD_sum+=np.sum(tree_dist[y_preds[:,i],label])        
    return AHD_sum/k

def top_k_HPK(label, score, k=5):  
    #label: [n], score: [n,n_classes]
    #global tree_dist
    y_preds=np.argsort(-score,axis=-1)[:,:k] #[n,k]
    HPK_sum=0.0
    hdist = np.sort(tree_dist[label,:],axis=-1)[:,1+k] #[n]?
    hCorrectlist=[np.where(tree_dist[label[j],:]<=hdist[j])[0] for j in range(len(label))]
    for j in range(len(label)):
        tmp_intersect=np.intersect1d(y_preds[j,:],hCorrectlist[j])
        HPK_sum+= len(tmp_intersect)#np.sum(tree_dist[y_preds[:,i],label])        
    return HPK_sum/k
           
    
def top_k_HSR(feats_n, labels, k=250):  
    #feats_n: [n_test, dim], labels: [n_test]
    #global tree_dist
    if params.distance=='arccos':
        tree_dist2 = (np.pi/2)*tree_dist/(params.beta+tree_dist) #[n_classes, n_classes], range: [0, np.pi/2]
        tree_sim = np.cos(tree_dist2)
    elif params.distance=='euc':
        tree_dist2 = (2**0.5)*tree_dist/(params.beta+tree_dist) #[n_classes, n_classes], range: [0, 2**0.5]
        tree_sim = 1-tree_dist2**2/2
    else:
        raise
     
    #Similarity based on features
    cos_feats = np.matmul(feats_n, feats_n.transpose()) #[n_test, n_test]
    knn_inds=np.argsort(-cos_feats,axis=-1)[:,1:k+1] #[n_test, k]
    
    labels = labels.astype(int)
    #Similarity based on labels (classes)
    sim_cls = tree_sim[labels,:][:,labels] #[n_test, n_test]
    max_sim_inds= np.argsort(-sim_cls,axis=-1)[:,1:k+1] #[n_test, k]
    
    sim_sum=np.zeros(shape=[len(feats_n),k]) #[n_test, k]
    sim_max=np.zeros(shape=[len(feats_n),k])
    for i in range(k):
        sim_sum[:,i] = tree_sim[labels, labels[knn_inds[:,i]]]
        sim_max[:,i] = tree_sim[labels, labels[max_sim_inds[:,i]]]
    sim_sum = np.cumsum(sim_sum,axis=-1) #[n_test, k]
    sim_max = np.cumsum(sim_max,axis=-1)
    HS_np = sim_sum/sim_max #[n_test, k]
    return HS_np #[n_test, k] #np.nanmean(HS_np)












'''Evaluate model'''
method = params.method 
dist = params.distance 
backbone = params.backbone 
rand_backbone=params.rand_backbone
seed = params.seed
#batch_size= params.batch_size
small = params.small
use_val = params.use_val
resize = params.resize
last=params.last
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
    model.EMA_marker = torch.ones(len(model.W)) #, dtype=torch.bool 
    model.EMA_marker.requires_grad=False
    
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
if last:
    model_PATH_ += '_last'
old_stdout = sys.stdout
log_file = open(model_PATH_.replace('models_'+DATA_TAG+'/','record/'+DATA_TAG+'_')+'_eval.txt','w')
sys.stdout = log_file

if last:
    model_PATH_ = model_PATH_.replace('_last','')



print('Method:',method)
if method in ['DR','normface']:
    print('Distance:',dist)
print('Backbone:',backbone)
print('Seed:',params.seed)
print('Small nabirds:',small)
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
        #print('   Stress (normalized):',stresses_[0],stresses_[-1])
        if ema:
            print('   (EMA is ignored when \'mds_W==True\'.)')
        print('   CORR:',CORR)
    print('Dynamic scale:',dynamic)
    if dynamic:
        print('Scale (initial):',model.scale)
    else:
        print('Scale:',model.scale)
print('\n\n')
sys.stdout.flush()



if use_val&(not last):
    model.load_state_dict(torch.load(model_PATH_+'_best.pth'))
else:
    model.load_state_dict(torch.load(model_PATH_+'.pth'))
model = model.cuda()
model.eval()

var_inter, var_intra, arccos_proto_W = get_discriminality(model, trainloader)

softmax = nn.Softmax(dim=1)


'''We use shortest path distance for hierarchical distance, and not lowest common ancestor (LCA)'''
#k-acc (Top k-error): correct if the tree class is among the top k classes with the highest confidence.
#HDM (The hierarchical distance of a mistake): shortest path distance between true and predicted classes
#HDM is eqivalent with AHC when we use shortest path distance.
#AHD (The average hierarchical distance of top-k): average shortest path distance between true and top-k predicted classes
#HP@k (from "DEVISE" Frome et al. paper): hierarchical precision at k
#HS@R (from Barz & Denzler): hierarchical similarity at R (the authors name it hierarchical precision at R, but it is not actual precision and the above HP@k has very similar name. We name differently to avoid confusion)
#AHS@250 ()


corrects2 = 0
corrects2_proto = 0
top5_corrects2=0
top5_corrects2_proto=0
HC_sum = 0 #Hierarchical cost (summation)
HC_sum_proto = 0 #Hierarchical cost (summation)
#HDM_sum=0
AHD_sum=0
AHD_sum_proto=0
HPK_sum=0
HPK_sum_proto=0
#HSR_sum=0
#HSR_sum_proto=0
h_corrects2 = 4*[0]
h_corrects2_proto = 4*[0]
cnt2=0
conf_np = np.ones([0])*np.nan
pred_np = np.ones([0])*np.nan
labels_np = np.ones([0])*np.nan
feats_n_np = np.ones([0, 128])*np.nan

for i, data in enumerate(testloader):
    # get the inputs; data is a list of [inputs, labels]
            
    inputs, labels = data
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    feats = model(inputs).detach() #torch.Size([32, 128])
    
        
    if method in ['DR','normface']:
        feats_n = feats/torch.norm(feats, p=2, dim =1,keepdim=True) #feats_n = F.normalize(feats)
        '''Based on model.W'''
        if ema:
            cos_sim = torch.mm(feats_n, F.normalize(model.W).t())
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
        score = torch.mm(feats, model.W.t())+model.b #[32, 48]: [batch_size, n_classes]
        feats_n = F.normalize(feats)
    else:
        raise        
    if clspri:
        score +=model.log_cls_prior
        score_proto +=model.log_cls_prior
    corrects2+=torch.sum(torch.argmax(score,dim=-1)==labels)
    if method in ['DR','normface']:
        corrects2_proto+=torch.sum(torch.argmax(score_proto,dim=-1)==labels)
    
    labels_np_ = labels.detach().cpu().numpy()
    pred_np_ = torch.argmax(score,dim=-1).detach().cpu().numpy()
    top5_corrects2+= top_k_correct(labels_np_,score.detach().cpu().numpy(), k=5)
    
    if method in ['DR','normface']:
        pred_proto_np = torch.argmax(score_proto,dim=-1).detach().cpu().numpy()
        top5_corrects2_proto+= top_k_correct(labels_np_,score_proto.detach().cpu().numpy(), k=5)
    
    HC_sum+=np.sum(tree_dist[pred_np_,labels_np_])
    AHD_sum+=top_k_AHD(labels_np_,score.detach().cpu().numpy(), k=5)
    if method in ['DR','normface']:
        HC_sum_proto+=np.sum(tree_dist[pred_proto_np,labels_np_])
        AHD_sum_proto+=top_k_AHD(labels_np_,score_proto.detach().cpu().numpy(), k=5)
        
    HPK_sum+=top_k_HPK(labels_np_,score.detach().cpu().numpy(), k=5)
    #HSR_sum+=top_k_HSR(labels_np_,score.detach().cpu().numpy(), k=5)
    if method in ['DR','normface']:
        HPK_sum_proto+=top_k_HPK(labels_np_,score_proto.detach().cpu().numpy(), k=5)
        #HSR_sum_proto+=top_k_HSR(labels_np_,score_proto.detach().cpu().numpy(), k=5)
        
    for j in range(len(h_corrects2)):
        _, h_labels_np = higher_cls(labels_np_,level=j)
        _, h_pred_np = higher_cls(pred_np_,level=j)
        if method in ['DR','normface']:
            _, h_pred_proto_np = higher_cls(pred_proto_np,level=j)
        #print(h_labels_np, h_pred_np, np.equal(h_labels_np,h_pred_np))
        h_corrects2[j]+= np.sum(np.equal(h_labels_np,h_pred_np))
        if method in ['DR','normface']:
            h_corrects2_proto[j]+= np.sum(np.equal(h_labels_np,h_pred_proto_np))
                                
    cnt2+=len(labels.detach().cpu().numpy())
    
    conf, _ = torch.max(softmax(score),dim=-1)
    conf_np = np.concatenate([conf_np, conf.detach().cpu().numpy()],axis=0)
    pred_np = np.concatenate([pred_np, torch.argmax(score,dim=-1).detach().cpu().numpy()],axis=0)
    labels_np = np.concatenate([labels_np, labels.cpu().numpy()],axis=0)
    feats_n_np = np.concatenate([feats_n_np, feats_n.cpu().numpy()],axis=0)


print('Accuracy (test):',(corrects2/cnt2).detach().cpu().numpy())
if method in ['DR','normface']:
    print('Accuracy (test, proto):',(corrects2_proto/cnt2).detach().cpu().numpy())
    
print('Top-5 accuracy (test):',top5_corrects2/cnt2)
if method in ['DR','normface']:
    print('Top-5 accuracy (test, proto):',top5_corrects2_proto/cnt2)
print()

print('AHC (test):',HC_sum/cnt2)
if method in ['DR','normface']:
    print('AHC (test, proto):',HC_sum_proto/cnt2)
    
print('AHD (k=5, test):',AHD_sum/cnt2)
if method in ['DR','normface']:
    print('AHD (k=5, test, proto):',AHD_sum_proto/cnt2)
print()

print('HP@5 (test):',HPK_sum/cnt2)
if method in ['DR','normface']:
    print('HP@5 (test, proto):',HPK_sum_proto/cnt2)
print()
sys.stdout.flush()


HS_np=top_k_HSR(feats_n_np, labels_np, k=250) #HS_np: [n_test, k]  #score.detach().cpu().numpy(), 
HSR_vals = np.nanmean(HS_np,axis=0) #[k]
#HSR = HSR_vals[-1] #np.nanmean(HS_np[:,-1])
AHSR = np.trapz(HSR_vals,dx=1/(250-1))
for k in [1, 50, 100, 150, 200, 250]:
    print('HS@{0} (test): {1}'.format(k,HSR_vals[k-1]))
print('AHS@250 (test):', AHSR)
'''if method in ['DR','normface']:
    HSR_proto=top_k_HSR(feats_n_np, labels_np, k=250, proto=True)
    print('HS@k (k=250, test, proto):', HSR_proto)'''
print()

for j in range(len(h_corrects2)-1,-1,-1):
    _,_, h_cls_list_unq = higher_cls(np.array([0]),j,True)
    print('level: {0}, number of classes: {1}'.format(j,len(h_cls_list_unq)))
    print('   Accuracy (test): {0}'.format(h_corrects2[j]/cnt2)) #'   Accuracy (test, level={0}): {1}'.format(j,h_corrects2[j]/cnt2))
    if method in ['DR','normface']:
        print('   Accuracy (test, proto): {0}'.format(h_corrects2_proto[j]/cnt2)) #'   Accuracy (test, level={0}, proto): {1}'.format(j,h_corrects2_proto[j]/cnt2))
print()        
sys.stdout.flush()








'''Confidence plot'''
from scipy.stats import spearmanr
def plot_conf_acc(conf, pred, labels, bins=10+ 1,mode='equal'):
    global method
    if mode=='sort':
        sort_ind = np.argsort(conf)
        split_ind = np.array_split(sort_ind,bins,axis=0)
        mns = [np.mean(conf[split_ind[i]]) for i in range(bins) if len(split_ind[i])>=1]
    elif mode=='equal':
        conf_ranges = np.linspace(0,1,bins+1)
        split_ind = [np.where((conf_ranges[i]<=conf)&(conf<conf_ranges[i+1]))[0] for i in range(bins)]
        mns = [(conf_ranges[i]+conf_ranges[i+1])/2 for i in range(bins) if len(split_ind[i])>=1]
    else:
        raise
    
    conf_avg = [np.mean(conf[split_ind[i]]) for i in range(bins) if len(split_ind[i])>=1]
    freqs = [len(split_ind[i])/len(conf) for i in range(bins) if len(split_ind[i])>=1]
    accs = [np.nanmean(pred[split_ind[i]]==labels[split_ind[i]]) for i in range(bins) if len(split_ind[i])>=1]
    
    conf_avg = np.array(conf_avg)
    freqs = np.array(freqs)
    accs = np.array(accs)
    
    print('ECE:',sum(freqs*np.abs(accs-conf_avg)))
    print('MCE:',np.max(np.abs(accs-conf_avg)))
    print(spearmanr(mns,accs))

plot_conf_acc(conf_np, pred_np, labels_np, bins=15, mode='equal') #bins=15+1
print('\n\n')
sys.stdout.flush()


'''Accuracies'''
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pandas as pd

classes = list(data_cls['Class name'])
classes = [cls.replace(' ','_') for cls in classes]

metric_results = precision_recall_fscore_support(labels_np, pred_np, labels=np.arange(len(classes)))
confusion_matrix_result = confusion_matrix(labels_np,pred_np, labels=np.arange(len(classes)))

diff_labels= np.setdiff1d(np.arange(len(classes)),np.unique(labels_np))
diff_pred= np.setdiff1d(np.arange(len(classes)),np.unique(pred_np))

confusion_matrix_prob = confusion_matrix_result/(np.sum(confusion_matrix_result,1)[:,None])
confusion_matrix_df=pd.DataFrame(confusion_matrix_result.transpose(),index=classes,columns=classes)
confusion_matrix_prob_df=pd.DataFrame(confusion_matrix_prob.transpose(),index=classes,columns=classes)

if len(diff_labels)>=1:
    metric_results[1][diff_labels]=np.nan
    metric_results[2][diff_labels]=np.nan
if len(diff_pred)>=1:
    metric_results[0][diff_pred]=np.nan
    metric_results[2][diff_pred]=np.nan
    
np.set_printoptions(precision=4,threshold=np.inf)
print('Precision:',metric_results[0],'\n')
print('Recall:',metric_results[1],'\n')
print('F-score:',metric_results[2],'\n')

pd.set_option('display.max_columns',100)
pd.set_option('display.width',120)

# Plot the normalized confusion matrix
'''fig=plt.figure(figsize=(10, 10), dpi= 100, facecolor='w', edgecolor='k')
plt.imshow(confusion_matrix_prob_df)
plt.colorbar()
plt.title('Normalized confusion matrix')
plt.xlabel('True label')
plt.xticks(np.arange(n_classes),labels=list(confusion_matrix_prob_df.columns.values),fontsize=8, rotation=90)
plt.ylabel('Predicted label')
plt.yticks(np.arange(n_classes),labels=list(confusion_matrix_prob_df.index),fontsize=8)
#plt.savefig('confusion_maxtirx_'+method,dpi=200)
plt.show()'''
print('\n')
sys.stdout.flush()

labels_np = labels_np.astype(int)
pred_np = pred_np.astype(int)

for j in range(len(h_corrects2)-1,0,-1):
    _,_, h_cls_list_unq = higher_cls(np.array([0]),j,True)
    
    _, h_labels_np = higher_cls(labels_np,level=j)
    _, h_pred_np = higher_cls(pred_np,level=j)
    #_, h_pred_proto_np = higher_cls(pred_proto_np,level=j)
    
    metric_results = precision_recall_fscore_support(h_labels_np, h_pred_np, labels=np.arange(len(h_cls_list_unq)))
    confusion_matrix_result = confusion_matrix(h_labels_np, h_pred_np, labels=np.arange(len(h_cls_list_unq)))

    diff_labels= np.setdiff1d(np.arange(len(h_cls_list_unq)),np.unique(h_labels_np))
    diff_pred= np.setdiff1d(np.arange(len(h_cls_list_unq)),np.unique(h_pred_np))
    
    del confusion_matrix_df, confusion_matrix_prob_df

    confusion_matrix_prob = confusion_matrix_result/(np.sum(confusion_matrix_result,1)[:,None])
    confusion_matrix_df=pd.DataFrame(confusion_matrix_result.transpose(),index=h_cls_list_unq,columns=h_cls_list_unq)
    confusion_matrix_prob_df=pd.DataFrame(confusion_matrix_prob.transpose(),index=h_cls_list_unq,columns=h_cls_list_unq)

    if len(diff_labels)>=1:
        metric_results[1][diff_labels]=np.nan
        metric_results[2][diff_labels]=np.nan
    if len(diff_pred)>=1:
        metric_results[0][diff_pred]=np.nan
        metric_results[2][diff_pred]=np.nan

    #np.set_printoptions(precision=4,threshold=np.inf)
    print('level: {0}, number of classes: {1}'.format(j,len(h_cls_list_unq)))
    print('   Precision:',metric_results[0],'\n')
    print('   Recall:',metric_results[1],'\n')
    print('   F-score:',metric_results[2],'\n')
    print('')
    sys.stdout.flush()


'''Make trees'''
from skbio import DistanceMatrix
from skbio.tree import nj

W = model.W/torch.norm(model.W, p=2, dim =1,keepdim=True)
cos_sim = torch.mm(W, W.t())
arccos = torch.acos(torch.clip(cos_sim,-1,1))
euc = (2*(1-torch.clip(cos_sim,-1,1)))**0.5 #Euclidean distance
euc = euc.detach().cpu().numpy()
arccos=arccos.detach().cpu().numpy()

proto = model.proto.detach().cpu().numpy()

cos_sim_proto = np.matmul(proto, proto.transpose())
arccos_proto = np.arccos(np.clip(cos_sim_proto,-1,1))
euc_proto = (2*(1-np.clip(cos_sim_proto,-1,1)))**0.5#Euclidean distance

classes = list(data_cls['Class name']) #Class name
#classes = [str(cls) for cls in classes]
classes = [cls.replace(' ','_') for cls in classes]
classes = [cls.replace('(','-') for cls in classes]
classes = [cls.replace(')','') for cls in classes]
classes = [cls.replace(',','_') for cls in classes]
classes = [cls.replace('/','_') for cls in classes]
classes = [cls.replace('\'','_') for cls in classes]
for i in range(len(arccos)):
    arccos[i,i]=0
    euc[i,i]=0
    arccos_proto[i,i]=0
    euc_proto[i,i]=0
    
if dist=='arccos':
    dm = DistanceMatrix(arccos, classes)
elif dist=='euc':
    dm = DistanceMatrix(euc, classes)
else:
    raise

tree = nj(dm)
#print(tree.ascii_art())

dm_tree_dist = DistanceMatrix(tree_dist, classes)
tree_True = nj(dm_tree_dist)

#print('Compare tree:',tree.compare_subsets(tree_True))
#print('Compare tree:',tree.compare_tip_distances(tree_True)) 
#print(tree_True)

from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor, DistanceMatrix, ParsimonyTreeConstructor
import itertools
#from TreeConstruction import DistanceTreeConstructor

calculator = DistanceCalculator('identity')
constructor = DistanceTreeConstructor(calculator, 'nj')

dm_ = DistanceMatrix(list(classes))
for i1, i2 in itertools.combinations(range(len(classes)), 2):
    dm_[i1, i2] = dm[i1, i2]
    
dm_True = DistanceMatrix(list(classes))
for i1, i2 in itertools.combinations(range(len(classes)), 2):
    dm_True[i1, i2] = dm_tree_dist[i1, i2]
    
upgma_tree=constructor.upgma(dm_)
nj_tree=constructor.nj(dm_)
print('UPGMA based tree:')
Phylo.draw_ascii(upgma_tree, column_width =120)
print('\n')
print('NJ based tree:')
Phylo.draw_ascii(nj_tree, column_width =120)
print('\n')
sys.stdout.flush()

True_upgma_tree=constructor.upgma(dm_True)
True_nj_tree=constructor.nj(dm_True)

Phylo.write(True_upgma_tree.as_phyloxml(), "True_upgma_tree.txt", "newick")
Phylo.write(True_nj_tree.as_phyloxml(), "True_nj_tree.txt", "newick")
    
from ete3 import Tree
Phylo.write(upgma_tree.as_phyloxml(), "tmp_upgma_tree.txt", "newick")
#Phylo.write(nj_tree.as_phyloxml(), "tmp_nj_tree.txt", "newick")

with open('True_upgma_tree.txt') as f:
    lines = f.readlines()
    #print(lines)
True_upgma_tree=Tree(lines[0],format=1) #str(True_nj)
#print()

with open('tmp_upgma_tree.txt') as f:
    lines = f.readlines()
    #print(lines)
tmp_upgma_tree=Tree(lines[0],format=1) #str(True_nj)
#print(tmp_upgma_tree) 



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
            #print(i, data_cls['Class'][i], tmp_corr[0], tmp_corr[1])
            corr_list.append(tmp_corr[0])
            p_value_list.append(tmp_corr[1])
        
        p_value_np =np.array(p_value_list)
        nn_ind=np.logical_not(np.isnan(p_value_np))
        p_value_np = p_value_np[nn_ind]
        return fisher_corr(corr_list), hmean(p_value_np)#combine_pvalues(p_value_list)[-1] #,method='stouffer'
    else: #Elementwise correlation
        dm1_vec=np.ones([0])*np.nan
        dm2_vec=np.ones([0])*np.nan
        
        for i in range(len(dm1)):
            for j in range(i+1,len(dm2)): 
                dm1_vec = np.concatenate([dm1_vec, np.array([dm1[i,j]])])
                dm2_vec = np.concatenate([dm2_vec, np.array([dm2[i,j]])])
        
        return spearmanr(dm1_vec, dm2_vec)   
    

def get_higher_discriminality(W,level=0):
    _,h_cls_ind,_=higher_cls(np.arange(len(W)),level,True,verbose=0)
    h_cls_ind = np.array(h_cls_ind)
    
    indexes = [np.where(h_cls_ind==i)[0] for i in np.unique(h_cls_ind)]
    proto_ = [np.mean(W[indexes[i]],axis=0,keepdims=True) for i in np.unique(h_cls_ind)]
    mu = np.concatenate(proto_, axis=0) #[48, 128] #Before normalization
    #####mu = mu/np.linalg.norm(mu,axis=-1,keepdims=True)
    mu_dot = np.matmul(mu, mu.transpose()) #[?,?]
    mu_sq = np.sum(mu**2,axis=1,keepdims=True) #[?,1]
    sum_inter = 2*mu_sq-2*mu_dot #[?, ?]

    var_inter = np.sum(sum_inter)/(len(mu)*(len(mu)-1))
    #print(var_inter)   
    W_sqs = [np.sum(W[indexes[i]]**2,axis=1,keepdims=True) for i in np.unique(h_cls_ind)] #Each element: [N_i,1]
    W_dots = [np.matmul(W[indexes[i]], mu[i, None].transpose()) for i in np.unique(h_cls_ind)] #Each element: [N_i,1]
    sum_intra = [np.sum(W_sqs[i]+mu_sq[i,None]-2*W_dots[i])/len(indexes[i]) for i in np.unique(h_cls_ind)] #Each element: [N_i,1]->1
    var_intra = np.sum(sum_intra)/len(mu)   
    return var_inter, var_intra

def get_acc_measures(W,level=0,distance='euc',each=True):
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
                #print('   ', len(intersect), len(indexes[i])-1)
                
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
                    
                    #print(P_1s[-1],RPs[-1],MAP_Rs[-1])
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

'''for i in range(4,-1,-1):
    var_inter, var_intra = get_higher_discriminality(W.detach().cpu().numpy(), level=i)
    print('   Discriminativity (W, level: {0}): {1}'.format(i,var_inter/var_intra))
    #print('   Level:',i,'Discriminativity:',var_inter/var_intra)
print()

for i in range(4,-1,-1):
    var_inter, var_intra = get_higher_discriminality(proto, level=i)
    print('   Discriminativity (proto, level: {0}): {1}'.format(i,var_inter/var_intra))
    #print('   Level:',i, 'Discriminativity:',var_inter/var_intra)
print()'''

print('Accuracy measures')
for i in range(3,-1,-1):
    P_1, RP, MAP_R = get_acc_measures(W.detach().cpu().numpy(),level=i,distance=params.distance,each=True)
    print('   (W,     level: {0}) P_1: {1}, RP: {2}, MAP_R: {3}'.format(i,P_1, RP, MAP_R))
    P_1, RP, MAP_R = get_acc_measures(proto,level=i,distance=params.distance,each=True)
    print('   (proto, level: {0}) P_1: {1}, RP: {2}, MAP_R: {3}'.format(i,P_1, RP, MAP_R))
print('\n\n')



if dist=='arccos':
    print('Mean correlation (W):',corr_from_DMs(arccos, tree_dist, classwise=True))
    print('Elementwise correlation (W):',corr_from_DMs(arccos, tree_dist, classwise=False),'\n')
    
    print('Mean correlation (proto):',corr_from_DMs(arccos_proto, tree_dist, classwise=True))
    print('Elementwise correlation (proto):',corr_from_DMs(arccos_proto, tree_dist, classwise=False),'\n')
elif dist=='euc':
    print('Mean correlation (W):',corr_from_DMs(euc, tree_dist, classwise=True))
    print('Elementwise correlation (W):',corr_from_DMs(euc, tree_dist, classwise=False),'\n')
    
    print('Mean correlation (proto):',corr_from_DMs(euc_proto, tree_dist, classwise=True))
    print('Elementwise correlation (proto):',corr_from_DMs(euc_proto, tree_dist, classwise=False),'\n')
else:
    raise

result=tmp_upgma_tree.compare(True_upgma_tree, unrooted=True)
#True_upgma_tree.compare(True_nj_tree, unrooted=True)
#True_upgma_tree.robinson_foulds(True_nj_tree,unrooted_trees=True)
print('Robinson-Foulds distances:', result['rf'], result['max_rf'], result['norm_rf'])

sys.stdout.flush()
    
sys.stdout = old_stdout
log_file.close()
        

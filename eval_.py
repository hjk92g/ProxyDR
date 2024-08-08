import json

with open('config.json', 'r') as file:
    config_info = json.load(file)

DATA_init = config_info["DATA_init"]
#DATA_init = '/DATA_init/' #Location where plankton datasets are located. For instance, "DATA_init+'plankton_data/MicroS/'" should be the path of the MicroS dataset.
FOLDER_init = config_info["FOLDER_init"]
#FOLDER_init = '/FOLDER_init/' #Location where this repogistory "ProxyDR" is located.

import argparse

parser = argparse.ArgumentParser(description= 'Evaluate models for plankton data')
parser.add_argument('--GPU'      , default=2, type=int,  help='GPU number')
parser.add_argument('--dataset'     , default='MicroS',        help='MicroS/MicroL/MesoZ')
parser.add_argument('--method'     , default='DR',        help='DR/normface/softmax')
parser.add_argument('--distance'     , default='euc',        help='euc/arccos')
parser.add_argument('--backbone'       , default='inception', help='backbone: inception / resnet50') 
parser.add_argument('--rand_backbone'       , action='store_true', help='randomly initialized backbone (instead of pretrained backbone)') 
parser.add_argument('--size_inform'     , action='store_true', help='use image size information for classification')

parser.add_argument('--seed'      , default=1, type=int,  help='seed for training and test data split')
#parser.add_argument('--batch_size'      , default=32, type=int,  help='batch for training (, validation) and test')
parser.add_argument('--use_val'      , action='store_true',  help='use validation set to find best model')
parser.add_argument('--last'      , action='store_true',  help='use the last (epoch) model for evaluattion')

parser.add_argument('--aug'       , action='store_true', help='use augmentation') 
parser.add_argument('--clspri'   , action='store_true',  help='use class prior probability')
parser.add_argument('--ema'   , action='store_true',  help='use exponential moving average (EMA) for proxy')
parser.add_argument('--alpha'      , default=0.01, type=float,   help='alpha for EMA')
parser.add_argument('--mds_W'   , action='store_true',  help='use MDS (multi-dimensional scaling) for proxy')
parser.add_argument('--beta'      , default=1.0, type=float,   help='parameter beta for distance transformation T (only used in MDS). T(d)=pi*d/(beta+2*d) or T(d)=d/(beta+d)')
parser.add_argument('--dynamic'   , action='store_true',  help='update scale factor for AdaCos/AdaDR')
parser.add_argument('--CORR'   , action='store_true',  help='CORR loss proposed by Barz and Denzler (2019). It requires --mds_W==True to be turned on. Otherwise, it will be ignored. --dynamic==True has no effect')

params= parser.parse_args()

#print(params.method, params.backbone, params.ema) #DR inception True






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

import matplotlib.pyplot as plt
#from sklearn.metrics import top_k_accuracy_score #Requires python version 3.7 or newer

    
if params.dataset=='MicroS': 
    DATA_PATH = DATA_init+'plankton_data/MicroS/'
    DATA_TAG = 'MicroS'
elif params.dataset=='MicroL': 
    DATA_PATH = DATA_init+'plankton_data/MicroL/'
    DATA_TAG = 'MicroL'
elif params.dataset=='MesoZ': #gray
    DATA_PATH = DATA_init+'plankton_data/MesoZ/'
    DATA_TAG = 'MesoZ'
else:
    print('Unknown plankton dataset!')
    raise
    
    
if params.dataset in ['MesoZ']:
    gray = True
else:
    gray = False
    
plk_info = pd.read_csv(FOLDER_init+'ProxyDR/'+params.dataset+'_info.csv')
plk_cls = pd.read_csv(FOLDER_init+'ProxyDR/'+params.dataset+'_cls.csv')
    
N= len(plk_info) #Number of images
if params.use_val:
    N_train = int(0.7*N) #70%
    N_val = int(0.1*N) #10%
    N_test = N-N_train-N_val #20%
else:
    N_train = int(0.9*N) #90%
    N_test = N-N_train #10%

n_classes = len(plk_cls) #Number of clasees



plk_cls2=plk_cls.copy()
pd.options.mode.chained_assignment = None

max_len=1
for i in range(len(plk_cls2)):
    tmp_Path = str(plk_cls['Path'][i]).replace(DATA_PATH,'')
    plk_cls2['Path'][i] = tmp_Path.split('/')
    if len(plk_cls2['Path'][i])>max_len:
        max_len=len(plk_cls2['Path'][i])

    
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

tree_dist = np.ones([len(plk_cls2),len(plk_cls2)])*np.nan
for i in range(len(plk_cls2)):
    for j in range(len(plk_cls2)):
        tree_dist[i,j] = calc_dist(plk_cls2['Path'][i], plk_cls2['Path'][j], mode='diff')

chk_depth=0
def higher_cls(inds, level=0, return_list=False, verbose=0):
    global chk_depth
    h_cls_list=[]
    for i in range(len(plk_cls2)):
        depth = len(plk_cls2['Path'][i])
        if depth>level:
            h_cls_list.append(plk_cls2['Path'][i][level])
        else:
            if chk_depth==0:
                print('Some label will be used instead of higher level class (due to \'depth<=level\')!\n   depth:',depth, 'level:',level)
            h_cls_list.append(plk_cls2['Path'][i][-1])
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

class PlanktonDataset(Dataset):
    def __init__(self, info_path, cls_path, transform=None, size_inform = False):
        self.plk_info = pd.read_csv(info_path)
        self.plk_cls = pd.read_csv(cls_path)
        self.transform = transform
        self.img_sz= np.ones([0,2])*np.nan
        self.size_inform=size_inform
        
    def __getitem__(self, index):
        img_path = self.plk_info.iloc[index,0]
        if gray:     
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')
        #img = io.imread(img_path)
        cls = self.plk_info.iloc[index,1]
        label = torch.tensor(int(self.plk_cls.loc[self.plk_cls['Class']==cls,'Number']))
        if self.size_inform:
            size_info=torch.tensor(TF.get_image_size(img))
        if self.transform is not None:
            img = self.transform(img)    
            
        self.img_sz=np.concatenate([self.img_sz, np.array(TF.get_image_size(img)).reshape([1,2])],axis=0)
        img = TF.resize(img, size=[128,128]) #size=[299,299]) #
        if self.size_inform:
            return (img, label, size_info)
        else:            
            return (img, label)

    def __len__(self):
        return len(self.plk_info)

plk_dataset = PlanktonDataset(info_path=FOLDER_init+'ProxyDR/'+params.dataset+'_info.csv', 
                              cls_path=FOLDER_init+'ProxyDR/'+params.dataset+'_cls.csv',transform=None, size_inform= params.size_inform)

if params.use_val:
    train_set, val_set, test_set = torch.utils.data.random_split(plk_dataset, lengths=[N_train, N_val, N_test], generator=torch.Generator().manual_seed(params.seed)) 
else:
    train_set, test_set = torch.utils.data.random_split(plk_dataset, lengths=[N_train, N_test], generator=torch.Generator().manual_seed(params.seed)) 

class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn, size_inform=False):
        self.dataset = dataset
        self.map = map_fn
        self.size_inform=size_inform

    def __getitem__(self, index):
        if self.size_inform:
            img, label, size_info = self.dataset[index]
            img = self.map(img)
            return (img, label, size_info)
        else:
            img, label= self.dataset[index]
            img = self.map(img)
            return (img, label)

    def __len__(self):
        return len(self.dataset)
    
AUG=params.aug
if AUG:
    #In eval, we don't use augmentation
    train_set_tf = MapDataset(train_set, transforms.ToTensor(), size_inform=params.size_inform)
else:
    train_set_tf = MapDataset(train_set, transforms.ToTensor(), size_inform=params.size_inform)
test_set_tf = MapDataset(test_set,  transforms.ToTensor(), size_inform=params.size_inform)


trainloader = DataLoader(train_set_tf, batch_size=32, shuffle=True)
trainloader2 = DataLoader(MapDataset(train_set, transforms.ToTensor(), size_inform=params.size_inform), batch_size=200, shuffle=True) #To analyze training
testloader = DataLoader(test_set_tf, batch_size=32)
testloader2 = DataLoader(MapDataset(test_set, transforms.ToTensor(), size_inform=params.size_inform), batch_size=200, shuffle=True) #To analyze training



'''Analyze training set'''
use_cuda = torch.cuda.is_available()
labels_np=np.ones(0)*np.nan
for i, data in enumerate(trainloader):
    # get the inputs; data is a list of [inputs, labels]
    if params.size_inform:
        inputs, labels, size_info = data
    else:
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

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def Inception3(method, size_inform=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=not params.rand_backbone)
    if gray:
        model.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
    else:
        model.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
    model.fc=torch.nn.Linear(in_features=2048, out_features=128, bias=True)  
    if size_inform:
        model.fc_size=torch.nn.Linear(in_features=2, out_features=128, bias=True)  
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

def ResNet50(method, size_inform=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=not params.rand_backbone)
    #model = torchvision.models.resnet50(pretrained=True, progress=True) 
    
    if gray:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    #model.conv1 = nn.Conv2d(1, model.inplanes, kernel_size=7, stride=2, padding=3, bias=False) 
    model.fc = nn.Linear(in_features=512 * 4, out_features=128, bias=True) #in_features=512 * block.expansion
    if size_inform:
        model.fc_size=torch.nn.Linear(in_features=2, out_features=128, bias=True)  
        
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
                feat_tmp=feats_n[arg_y][dup_l[i]:dup_l[i]+dup_counts[i]].mean(dim=0)

                feats_n2[dup_l2[i]] = F.normalize(feat_tmp,dim=0) #feat_tmp/(torch.norm(feat_tmp, p=2)+ 0.00001) #Use average to update proxy when there are multiple points with the same class
        else:
            feats_n2 = feats_n[arg_y]            

        tmp_EMA_marker=model_.EMA_marker[y2]
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
    arccos_W_W = torch.acos(0.999999*torch.clip(cos_sim_W_W,-1,1)) #[n_classes, n_classes]
    arccos_W_W[torch.arange(len(WN),dtype=int), torch.arange(len(WN),dtype=int)] = np.nan #torch.nan
    
    arccos_feat_W, I, arccos_W_W = arccos_feat_W.cpu().numpy(), I.cpu().numpy(), arccos_W_W.cpu().numpy()
    arccos_feat_W_df = arccos_feat_W_df.cpu().numpy()
    return arccos_feat_W, arccos_feat_W_df, I, arccos_W_W
    
def get_discriminality(model, loader, size_inform=False):
    global use_cuda 
    #Related papers "Negative Margin Matters: Understanding Margin in Few-shot Classification" (www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490426.pdf)
    # and "Revisiting Training Strategies and Generalization Performance in Deep Metric Learning" (proceedings.mlr.press/v119/roth20a/roth20a.pdf)
    feats_np = np.ones([0,128])*np.nan
    labels_np = np.ones([0])*np.nan
    model.eval()
    for i, data in enumerate(loader):
        # get the inputs; data is a list of [inputs, labels]
        if size_inform:
            inputs, labels, size_info = data
            if use_cuda:
                inputs, labels, size_info = inputs.cuda(), labels.cuda(), size_info.cuda()
        else:
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

        feats = model(inputs) #torch.Size([32, 128])        
        if size_inform:
            feats_size=model.fc_size(torch.log(size_info))
            feats+=feats_size
        
        feats_n = F.normalize(feats).detach()
        feats_np = np.concatenate([feats_np, feats_n.cpu().numpy()],axis=0)
        labels_np = np.concatenate([labels_np, labels.cpu().numpy()],axis=0)
        
    indexes = [np.where(labels_np==i)[0] for i in range(len(plk_cls))]
    proto_ = [np.mean(feats_np[indexes[i]],axis=0,keepdims=True) for i in range(len(plk_cls))]
    mu = np.concatenate(proto_, axis=0) #[48, 128] #Before normalization
    proto_ = [proto_[i]/(np.linalg.norm(proto_[i],axis=-1,keepdims=True)+1e-12) for i in range(len(plk_cls))]
    proto = np.concatenate(proto_, axis=0) #[48, 128]
    #model.proto_nan = torch.tensor(proto,dtype=torch.float,device = model.W.device) #With NaNs
    np.random.seed(params.seed)
    for i in range(len(plk_cls)):
        if len(indexes[i])==0:
            feats_rand = np.random.normal(size=128)
            proto[i,:] = feats_rand/(np.linalg.norm(feats_rand,keepdims=True)+1e-12)
    
    model.proto = torch.tensor(proto,dtype=torch.float,device = model.W.device) #Without NaNs

    WN = F.normalize(model.W.detach())
    W_np = WN.cpu().numpy()
    
    mu_dot = np.matmul(mu, mu.transpose()) #[48,48]
    mu_sq = np.sum(mu**2,axis=1,keepdims=True) #[48,1]
    sum_inter = 2*mu_sq-2*mu_dot #[48, 48]
    
    var_inter = np.sum(sum_inter)/(len(mu)*(len(mu)-1))
    
    feat_sqs = [np.sum(feats_np[indexes[i]]**2,axis=1,keepdims=True) for i in range(len(plk_cls))] #Each element: [N_i,1]
    feat_dots = [np.matmul(feats_np[indexes[i]], mu[i, None].transpose()) for i in range(len(plk_cls))] #Each element: [N_i,1]
    sum_intra = [np.sum(feat_sqs[i]+mu_sq[i,None]-2*feat_dots[i])/len(indexes[i]) for i in range(len(plk_cls))] #Each element: [N_i,1]->1
    var_intra = np.sum(sum_intra)/len(mu)
    
    
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
size_inform = params.size_inform
seed = params.seed
#batch_size= params.batch_size
use_val = params.use_val
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
    model = Inception3(method=method, size_inform = size_inform)
elif backbone == 'resnet50':
    model = ResNet50(method=method, size_inform = size_inform)
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
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5, weight_decay=0.) 
    if dynamic:
        optimizer0 = torch.optim.Adam([model.log_scale],lr=1e-5, weight_decay=0.) 
else:
    raise
    
criterion = nn.CrossEntropyLoss() 
use_cuda = torch.cuda.is_available()
losses_ = []

model_PATH_ = 'models_'+DATA_TAG+'/' #'models_MicroS/'
if use_val:
    model_PATH_ += backbone+'_'+method+'_vsd'+str(seed)
else:
    model_PATH_ += backbone+'_'+method+'_sd'+str(seed)
    
if rand_backbone:
    model_PATH_ = model_PATH_.replace(backbone,'rd'+backbone)
    
if size_inform:
    model_PATH_ = model_PATH_.replace(backbone,backbone+'_size')
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
print('Size informed:',size_inform)
print('Seed:',params.seed)
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

#print(model_PATH_) 

if use_val&(not last):
    model.load_state_dict(torch.load(model_PATH_+'_best.pth'))
else:
    model.load_state_dict(torch.load(model_PATH_+'.pth'))
model = model.cuda()
model.eval()

var_inter, var_intra, arccos_proto_W = get_discriminality(model, trainloader, size_inform=size_inform)

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
h_corrects2 = 5*[0]
h_corrects2_proto = 5*[0]
cnt2=0
conf_np = np.ones([0])*np.nan
conf_proto_np = np.ones([0])*np.nan
pred_np = np.ones([0])*np.nan
pred_proto_np_ = np.ones([0])*np.nan
labels_np = np.ones([0])*np.nan
feats_n_np = np.ones([0, 128])*np.nan

for i, data in enumerate(testloader):
    # get the inputs; data is a list of [inputs, labels]
    if size_inform:
        inputs, labels, size_info = data
        if use_cuda:
            inputs, labels, size_info = inputs.cuda(), labels.cuda(), size_info.cuda()
    else:
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

    feats = model(inputs).detach() #torch.Size([32, 128])
    
    if size_inform:
        feats_size=model.fc_size(torch.log(size_info))
        feats+=feats_size
        feats = feats.detach()
    
    if method in ['DR','normface']:
        feats_n = F.normalize(feats)
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
        score = torch.mm(feats, model.W.t())+model.b #[batch_size, n_classes]
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
                                
    #corrects2+=torch.sum(torch.argmin(arccos,dim=-1)==labels)
    cnt2+=len(labels.detach().cpu().numpy())
    
    conf, _ = torch.max(softmax(score),dim=-1)
    conf_np = np.concatenate([conf_np, conf.detach().cpu().numpy()],axis=0)
    pred_np = np.concatenate([pred_np, torch.argmax(score,dim=-1).detach().cpu().numpy()],axis=0)
    labels_np = np.concatenate([labels_np, labels.cpu().numpy()],axis=0)
    feats_n_np = np.concatenate([feats_n_np, feats_n.cpu().numpy()],axis=0)
    if method in ['DR','normface']:
        conf_proto, _ = torch.max(softmax(score_proto),dim=-1)
        conf_proto_np = np.concatenate([conf_proto_np, conf_proto.detach().cpu().numpy()],axis=0)
        pred_proto_np_ = np.concatenate([pred_proto_np_, torch.argmax(score_proto,dim=-1).detach().cpu().numpy()],axis=0)



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
    print('   Accuracy (test): {0}'.format(h_corrects2[j]/cnt2)) 
    if method in ['DR','normface']:
        print('   Accuracy (test, proto): {0}'.format(h_corrects2_proto[j]/cnt2)) 
print()        
sys.stdout.flush()


'''Confidence plot'''
from scipy.stats import spearmanr, pointbiserialr
del conf, labels
def plot_conf_acc(conf, pred, labels, bins=10+ 1, mode='equal', proto=False):
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
    
    if proto:
        print('ECE (proto):',sum(freqs*np.abs(accs-conf_avg)))
        print('MCE (proto):',np.max(np.abs(accs-conf_avg)))
        print('(proto):',spearmanr(mns,accs))
        pb_corr = pointbiserialr(pred==labels,conf)
        print('Point biserial corr (proto):',pb_corr[0],pb_corr[1])
    else:
        print('ECE:',sum(freqs*np.abs(accs-conf_avg)))
        print('MCE:',np.max(np.abs(accs-conf_avg)))
        print(spearmanr(mns,accs))
        pb_corr = pointbiserialr(pred==labels,conf)
        print('Point biserial corr:',pb_corr[0],pb_corr[1])
        

plot_conf_acc(conf_np, pred_np, labels_np, bins=15, mode='equal') #bins=15+1
if method in ['DR','normface']:
    print()
    plot_conf_acc(conf_proto_np, pred_proto_np_, labels_np, bins=15, mode='equal', proto=True) #bins=15+1
print('\n\n')
sys.stdout.flush()


'''Accuracies'''
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pandas as pd

classes = list(plk_cls['Class'])
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

for j in range(len(h_corrects2)-1,-1,-1):
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

classes = list(plk_cls['Class'])
classes = [cls.replace(' ','_') for cls in classes]
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
Phylo.draw_ascii(upgma_tree)
print('\n')
print('NJ based tree:')
Phylo.draw_ascii(nj_tree)
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

def corr_from_DMs(dm1, dm2, whole=True, classwise=True):
    if classwise:
        corr_list=[]
        p_value_list=[]
        if whole:
            for i in range(len(dm1)):
                tmp_corr = spearmanr(dm1[:,i], dm2[:,i])
                #print(i, plk_cls['Class'][i], tmp_corr[0], tmp_corr[1])
                corr_list.append(tmp_corr[0])
                p_value_list.append(tmp_corr[1])
        else:            
            if params.dataset=='MicroS': #Ignore nonliving
                for i in range(0,89+1): #Only living classes 
                    tmp_corr = spearmanr(dm1[:89+1,i], dm2[:89+1,i]) 
                    corr_list.append(tmp_corr[0])
                    p_value_list.append(tmp_corr[1])            
            elif params.dataset=='MicroL': #Mesozooplankton and microplankton (ignore nonplankton)
                for i in range(0,75+1): 
                    tmp_corr = spearmanr(dm1[:75+1,i], dm2[:75+1,i]) 
                    corr_list.append(tmp_corr[0])
                    p_value_list.append(tmp_corr[1])
            elif params.dataset=='MesoZ': #Mesozooplankton (ignore artifacts, protists, debris and pom)
                for i in range(0,29+1): #Only mesozooplankton classes
                    tmp_corr = spearmanr(dm1[:29+1,i], dm2[:29+1,i]) 
                    corr_list.append(tmp_corr[0])
                    p_value_list.append(tmp_corr[1])
            else:
                print('eval_.py code is not implemented for this dataset! Try other python codes!')
                raise
        return fisher_corr(corr_list), hmean(p_value_list)#combine_pvalues(p_value_list)[-1] #,method='stouffer'
    else: #Elementwise correlation
        dm1_vec=np.ones([0])*np.nan
        dm2_vec=np.ones([0])*np.nan
        if whole:
            for i in range(len(dm1)):
                for j in range(i+1,len(dm2)): 
                    dm1_vec = np.concatenate([dm1_vec, np.array([dm1[i,j]])])
                    dm2_vec = np.concatenate([dm2_vec, np.array([dm2[i,j]])])
        else:            
            if params.dataset=='MicroS': #Ignore nonliving
                for i in range(0,89+1): #Only living classes
                    for j in range(i+1,89+1): 
                        dm1_vec = np.concatenate([dm1_vec, np.array([dm1[i,j]])])
                        dm2_vec = np.concatenate([dm2_vec, np.array([dm2[i,j]])])
            elif params.dataset=='MicroL': #Mesozooplankton and microplankton (ignore nonplankton)
                for i in range(0,75+1): 
                    for j in range(i+1,75+1): 
                        dm1_vec = np.concatenate([dm1_vec, np.array([dm1[i,j]])])
                        dm2_vec = np.concatenate([dm2_vec, np.array([dm2[i,j]])])
            elif params.dataset=='MesoZ': #Mesozooplankton (ignore artifacts, protists, debris and pom)
                for i in range(0,29+1): #Only mesozooplankton classes
                    for j in range(i+1,29+1): 
                        dm1_vec = np.concatenate([dm1_vec, np.array([dm1[i,j]])])
                        dm2_vec = np.concatenate([dm2_vec, np.array([dm2[i,j]])])
            else:
                print('eval_.py code is not implemented for this dataset! Try other python codes!')
                raise
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
for i in range(4,-1,-1):
    P_1, RP, MAP_R = get_acc_measures(W.detach().cpu().numpy(),level=i,distance=params.distance,each=True)
    print('   (W,     level: {0}) P_1: {1}, RP: {2}, MAP_R: {3}'.format(i,P_1, RP, MAP_R))
    P_1, RP, MAP_R = get_acc_measures(proto,level=i,distance=params.distance,each=True)
    print('   (proto, level: {0}) P_1: {1}, RP: {2}, MAP_R: {3}'.format(i,P_1, RP, MAP_R))
print('\n\n')



if dist=='arccos':
    print('Mean correlation (W, whole):',corr_from_DMs(arccos, tree_dist, whole=True, classwise=True))
    print('Elementwise correlation (W, whole):',corr_from_DMs(arccos, tree_dist, whole=True, classwise=False),'\n')

    print('Mean correlation (W, Living):',corr_from_DMs(arccos, tree_dist, whole=False, classwise=True))
    print('Elementwise correlation (W, Living):',corr_from_DMs(arccos, tree_dist, whole=False, classwise=False),'\n')
    
    print('Mean correlation (proto, whole):',corr_from_DMs(arccos_proto, tree_dist, whole=True, classwise=True))
    print('Elementwise correlation (proto, whole):',corr_from_DMs(arccos_proto, tree_dist, whole=True, classwise=False),'\n')

    print('Mean correlation (proto, Living):',corr_from_DMs(arccos_proto, tree_dist, whole=False, classwise=True))
    print('Elementwise correlation (proto, Living):',corr_from_DMs(arccos_proto, tree_dist, whole=False, classwise=False),'\n')
elif dist=='euc':
    print('Mean correlation (W, whole):',corr_from_DMs(euc, tree_dist, whole=True, classwise=True))
    print('Elementwise correlation (W, whole):',corr_from_DMs(euc, tree_dist, whole=True, classwise=False),'\n')

    print('Mean correlation (W, Living):',corr_from_DMs(euc, tree_dist, whole=False, classwise=True))
    print('Elementwise correlation (W, Living):',corr_from_DMs(euc, tree_dist, whole=False, classwise=False),'\n')
    
    print('Mean correlation (proto, whole):',corr_from_DMs(euc_proto, tree_dist, whole=True, classwise=True))
    print('Elementwise correlation (proto, whole):',corr_from_DMs(euc_proto, tree_dist, whole=True, classwise=False),'\n')

    print('Mean correlation (proto, Living):',corr_from_DMs(euc_proto, tree_dist, whole=False, classwise=True))
    print('Elementwise correlation (proto, Living):',corr_from_DMs(euc_proto, tree_dist, whole=False, classwise=False),'\n')
else:
    raise

result=tmp_upgma_tree.compare(True_upgma_tree, unrooted=True)
#True_upgma_tree.compare(True_nj_tree, unrooted=True)
#True_upgma_tree.robinson_foulds(True_nj_tree,unrooted_trees=True)
print('Robinson-Foulds distances:', result['rf'], result['max_rf'], result['norm_rf'])

sys.stdout.flush()
    
sys.stdout = old_stdout
log_file.close()
        

import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms


from models.network import *
from models.dataloader import *

# fix seeds
def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# test model
def test_model(args,epoch, model, valid_dataloader, criterion):
    
    model.eval()

    total_loss = 0.0    
    
    with torch.no_grad():
        for i, (curr_img, next_img, rel_pose_gt) in tqdm(enumerate(valid_dataloader, 0)):

            curr_img, next_img, rel_pose_gt = curr_img.cuda(), next_img.cuda(), rel_pose_gt.cuda() 
            outputs = model(curr_img, next_img)

            loss = criterion(outputs,rel_pose_gt)
            total_loss += loss.item()
        
    print('Test: epoch: %d loss: %.6f ' %(epoch, total_loss))
    return(total_loss)

    
def train_model(args, epoch, model, train_dataloader, criterion):  # train model
    
    model.train()
    
    total_loss = float(0.0)    

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = 0)
    
    for i, (curr_img, next_img, rel_pose_gt) in tqdm(enumerate(train_dataloader,0)):
        
        rel_pose_gt = rel_pose_gt.cuda() 
        curr_img, next_img = curr_img.cuda(), next_img.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        
        outputs = model(curr_img, next_img)
        if args.model == "InceptionV3":
            outputs = outputs[0]
        
        loss = criterion(outputs, rel_pose_gt)
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        
    
    # loss and acc
    print('Train: epoch: %d loss: %.6f' %(epoch, total_loss))
    return

''' 

versions
Base model
      Archi                 lr            BS              remarks
1)    ResNet18          0.000001         128              no augmentations
1.2)  ResNet18          0.000001         128              balanced loss
2)    InceptionV3
'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SimCol_3D_Pose_Estimation')
    
    # Model and version parameters
    parser.add_argument('--model',                  type=str,   default='ResNet18SimpleCBS',      help='Feature extraction model')
    parser.add_argument('--checkpoint',             type=str,   default='R18SimpleCBS_12_LOG',           help='path to checkpoint, None if none.')

    
    # Training parameters
    parser.add_argument('--epochs',                 type=int,   default=101,                help='number of epochs to train')
    parser.add_argument('--batch_size',             type=int,   default=64,                help='batch_size')
    parser.add_argument('--lr',                     type=float, default=0.0000075,           help='0.00001')
    parser.add_argument('--print_freq',             type=int,   default=1,                  help='print frequency.')    
    
    # CBS
    parser.add_argument('--use_cbs',            type=bool,      default=True,       help='use CBS')
    parser.add_argument('--std',                type=float,     default=1.0,         help='')
    parser.add_argument('--std_factor',         type=float,     default=0.9,         help='')
    parser.add_argument('--cbs_epoch',          type=int,       default=10,           help='')
    parser.add_argument('--kernel_size',        type=int,       default=3,           help='')
    parser.add_argument('--fil1',               type=str,       default='LOG',       help='gau, LOG')
    

    args = parser.parse_args()
    print(args.model)


    # Set random seed
    seed_everything()  
    
    '''dataset preparation'''
    # train and test dataloader

    # train
    train_subsets = ['../dataset/Train/Input/SyntheticColon_I_Train', '../dataset/Train/Input/SyntheticColon_II_Train']
    train_subjects = [['S1', 'S2','S3', 'S6', 'S7', 'S8', 'S11', 'S12', 'S13'], ['B1', 'B2','B3', 'B6', 'B7', 'B8', 'B11', 'B12', 'B13']]
    # val
    val_subsets = ['../dataset/Val/Input/SyntheticColon_I_Val', '../dataset/Val/Input/SyntheticColon_II_Val']
    val_subjects = [['S4', 'S9', 'S14'], ['B4', 'B9', 'B14']]


    transform = transforms.Compose([
                transforms.Resize((300,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    ViTDataset = False
    # model
    if args.model == 'ResNet18':
        model = R18Pretrained()
    elif args.model == 'ResNet18CBS':
        model = R18CBS(args)
        model.model.get_new_kernels(0)
    elif args.model == 'ResNet18SimpleCBS':
        model = R18SimpleCBS(args)
        model.get_new_kernels(0)
    
    elif args.model == 'ResNet50':
        model = R50Pretrained()
    elif args.model == 'InceptionV3':
        model = InceptionV3Pretrained()
        transform = transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])
    elif args.model == 'ViT':
        model = ViTPretrained()
        transform = None
        ViTDataset = True
    model.cuda()

    # train_dataset
    train_dataset = SimCol3DDataloader(train_subsets, train_subjects, ViT = ViTDataset, transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)

    # Val_dataset
    val_dataset = SimCol3DDataloader(val_subsets, val_subjects, ViT = ViTDataset, transform=transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size*2, shuffle=False, num_workers=8)

    # loss criterion
    criterion = nn.MSELoss()
    
    best_epoch = [0]
    best_results = [10000.0]
    
    for epoch in range(1, args.epochs):

        if args.use_cbs:
            if args.model == 'ResNet18CBS':
                model.model.get_new_kernels(epoch)
            elif args.model == 'ResNet18SimpleCBS':
                model.get_new_kernels(0)
            model.cuda()
        
        train_model(args, epoch, model, train_dataloader, criterion)
        test_loss = test_model(args, epoch, model, val_dataloader, criterion)
    
        print(test_loss)
        if test_loss <= best_results[0]:
            best_results[0] = test_loss
            best_epoch[0] = epoch

            checkpoint = {'lr': args.lr, 'b_s': args.batch_size, 'state_dict': model.state_dict() }
            torch.save(checkpoint, os.path.join('checkpoints',(args.checkpoint + 'best_model.pth')))
        
        print('Best epoch: %d | Best loss: %.6f' %(best_epoch[0], best_results[0]))
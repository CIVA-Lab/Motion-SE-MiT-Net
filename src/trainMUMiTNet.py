import argparse
import os
import warnings
import torch
import torch.optim as optim
import glob
import torch.hub
import pandas as pd

from torchsummary import summary
from torch.optim import lr_scheduler

from data.dataLoader import (CustomMotionPathLoader, MUSENetDataLoader)
from nets.MUMiTNet import MUMiTNet
from trainers.trainer import trainMSModel
import segmentation_models_pytorch as smp


warnings.filterwarnings("ignore")
# parser getting parameters from caller
parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--dataset',
                    help='The CTC Dataset (needs to be inside the `dataset/train` directory.)')
parser.add_argument('--resize_to', type=int, nargs=2,
                    help='Resize datasets with non homogenous image size (e.g. Fluo-C2DL-MSC) to a similar size.')
parser.add_argument('--train_resolution', type=int, nargs=2, default=[512, 512],
                    help='Training patch resolution.')

args = parser.parse_args()

dataset = args.dataset
basedir = os.path.join('./dataset/train/', dataset)

# model name
modelName = dataset + '-MUMiTNet'

print("**********************")
print(modelName)
print("**********************")

train_resolution = args.train_resolution

# get video path from Flist.txt
fileName = './files/seqNum.txt'
df = pd.read_csv(fileName, names=['filename'])
nameListTrain = df['filename']

trainDataset = []
valDataset =[]

for seqPath in nameListTrain:
    print(seqPath)
    
    # inputs, labels and markers path
    folderData = sorted(glob.glob(basedir + seqPath + "/*.tif"))
    folderBGS = sorted(glob.glob(basedir + seqPath + "_BGS/*.png"))
    folderFlux = sorted(glob.glob(basedir + seqPath + "_FLUX/*.png"))
    folderMask = sorted(glob.glob(basedir + seqPath +  "_ST/BSEG/*.png")) 
    folderMarker = sorted(glob.glob(basedir + seqPath +  "_ST/MARKER/*.png")) 
    
    folderData = folderData[3:-1]
    folderBGS = folderBGS[2:-1]
    folderMask = folderMask[3:-1]
    folderMarker = folderMarker[3:-1]
    
    print(folderData[0])
    print(folderBGS[0])
    print(folderFlux[0])
    print(folderMask[0])
    print(folderMarker[0])

    print(folderData[-1])
    print(folderBGS[-1])
    print(folderFlux[-1])
    print(folderMask[-1])
    print(folderMarker[-1])
    
    dataset = CustomMotionPathLoader(folderData, folderBGS, folderFlux,  folderMask, folderMarker)
    
    trLengths = int(len(dataset)*0.9);
    lengths = [trLengths, len(dataset) - trLengths]
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    
    trainDataset.extend(train_dataset)
    valDataset.extend(val_dataset)

    del dataset
    del train_dataset
    del val_dataset
    
print(len(trainDataset))
print(len(valDataset))

impaths = []
bgspaths = []
fluxpaths = []
maskpaths = []
markerpaths = []

for trainData in trainDataset:
    tData, tBGS, tFlux, tMask, tMarker = trainData
    
    impaths.append(tData)
    bgspaths.append(tBGS)
    fluxpaths.append(tFlux)
    maskpaths.append(tMask)
    markerpaths.append(tMarker)

   
train_df = pd.DataFrame({
    'image': impaths,
    'bgs': bgspaths,
    'flux': fluxpaths,
    'label': maskpaths,
    'marker': markerpaths
})

trainset = MUSENetDataLoader(train_df, resolution = train_resolution, resize_to=args.resize_to)

del impaths
del bgspaths
del fluxpaths
del maskpaths
del markerpaths

impaths = []
bgspaths = []
fluxpaths = []
maskpaths = []
markerpaths = []

for valData in valDataset:
    vData, vBGS, vFlux, vMask, vMarker = valData
    
    impaths.append(vData)
    bgspaths.append(vBGS)
    fluxpaths.append(vFlux)
    maskpaths.append(vMask)
    markerpaths.append(vMarker)

   
val_df = pd.DataFrame({
    'image': impaths,
    'bgs': bgspaths,
    'flux': fluxpaths,
    'label': maskpaths,
    'marker': markerpaths
})

valset = MUSENetDataLoader(val_df, resolution = train_resolution, resize_to=args.resize_to)

del impaths
del bgspaths
del fluxpaths
del maskpaths
del markerpaths

del trainDataset
del valDataset

# set batch size
batchSize = 8

trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
valLoader = torch.utils.data.DataLoader(valset, batch_size=batchSize, shuffle=True)

dataLoaders = {
    'train': trainLoader,
    'val': valLoader
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# pretrained se-resnet50 on ImageNet
model_class = smp.Unet(encoder_name='mit_b5', encoder_weights='imagenet', classes=1, activation=None)
encoder = model_class.encoder

numClass = 2
model = MUMiTNet(numClass, encoder).to(device)

## summarize the network
#summary(model, [(3, train_resolution[1], train_resolution[0]), (3, train_resolution[1], train_resolution[0])])

# using Adam optimizer with learning rate 1e-4
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# decrease learning rate by 0.5 after each 40th epoch
lrScheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
# train model with 100 epoch
trainMSModel(dataLoaders, model, optimizer, lrScheduler, earlyStopNumber=100, numEpochs=600, modelN = modelName)



    

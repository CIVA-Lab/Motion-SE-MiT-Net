import argparse
import os
import glob
import torch.nn.functional as F
from warnings import filterwarnings

import numpy as np
import pandas as pd
import skimage.io as sio
import torch
import time
import imageio
from datetime import timedelta
from torchvision import transforms

from data.dataLoader import (CustomMotionTestPathLoader, MUSENetTestDataLoader)
from nets.MUNet import MUNet
from postproc import postprocess_mask_and_markers

filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(
        description='Generate a segmentation for a given CTC dataset.')
    parser.add_argument('--dataset',
                        help='The CTC Dataset (needs to be inside the `trainsets` directory.)')
    parser.add_argument('--sequence_id',
                        help='The sequence/video ID. (e.g. `01`, `02`)')
    parser.add_argument('--datatype',
                        help='Choose data type if it is train or test. (e.g. `train`, `test`)')
    parser.add_argument('--area_threshold', type=int, default=200,
                        help='Threshold area under which detections are disregared (considered spurious).')

    args = parser.parse_args()
    
    dataset = args.dataset
    data_type = args.datatype
    basedir = os.path.join('./dataset/' + data_type + '/', dataset)
    sequence_id = args.sequence_id
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    networkName = 'MUNet'
    modelName = dataset + '-' + networkName
    
    numClass = 2
    model = MUNet(numClass).to(device)
    
    # load trained model
    model.load_state_dict(torch.load('./models/' + modelName + '.pt'))
    model.eval() 
    
    folderData = sorted(glob.glob(basedir + "/" + sequence_id + "/*.jpg"))
    folderBGS = sorted(glob.glob(basedir + "/" + sequence_id + "_BGS/*.png"))
    folderFlux = sorted(glob.glob(basedir + "/" + sequence_id + "_FLUX/*.png"))
    
    folderData = folderData[3:-1]
    folderBGS = folderBGS[2:-1]
    
    print(folderData[0])
    print(folderBGS[0])
    print(folderFlux[0])

    print(folderData[-1])
    print(folderBGS[-1])
    print(folderFlux[-1])
    
    testDataset = CustomMotionTestPathLoader(folderData, folderBGS, folderFlux)
    
    print(len(testDataset))

    impaths = []
    bgspaths = []
    fluxpaths = []

    for testData in testDataset:
        tData, tBGS, tFlux = testData
        
        impaths.append(tData)
        bgspaths.append(tBGS)
        fluxpaths.append(tFlux)
    
    evalset = MUSENetTestDataLoader(pd.DataFrame({
        'image': impaths,
        'bgs': bgspaths,
        'flux': fluxpaths,
    }))
    
    # set mask path
    maskDir = os.path.join('./output/' + data_type + '/' + networkName + "/" + dataset + "/" + sequence_id + '/Mask/')
    # create path if not exist
    if not os.path.exists(maskDir):
        os.makedirs(maskDir)
    
    # set mask path
    markerDir = os.path.join('./output/' + data_type + '/' + networkName + "/" + dataset + "/" + sequence_id + '/Marker/')
    # create path if not exist
    if not os.path.exists(markerDir):
        os.makedirs(markerDir)
        
    # set mask path
    outDir = os.path.join('./output/' + data_type + '/' + networkName + "/" + dataset + "/" + sequence_id + '/Label')
    # create path if not exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        
    
    # start timer 
    startTime = time.time()
    
    for i in range(len(evalset)):
        p = evalset.impaths[i]
        
        img1, img2 = evalset[i]

        H, W = sio.imread(evalset.impaths[i]).shape
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((H, W)),
            transforms.ToTensor()
        ])
        
        # Predict
        pred = model(img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device))
        
        # The loss functions include the sigmoid function.
        pred = F.sigmoid(pred)
        pred = pred.squeeze()
        pred = pred.detach().cpu()
        
        mask, markers = pred
        
        mask = transform(mask).squeeze().numpy()
        markers = transform(markers).squeeze().numpy()

        labeled_mask = postprocess_mask_and_markers(mask, markers, area_thresh=args.area_threshold)
        labeled_mask = labeled_mask.astype('uint16')

        outname = p.split('/')[-1]
        outpath = f'{outDir}/mask{outname[1:]}'
        print(f'Saving to: {outpath} ...')
        sio.imsave(outpath, labeled_mask)
        
        outPredMaskNorm = 255 * mask
        outPredMaskUint8 = outPredMaskNorm.astype(np.uint8)
  
        outPredMarkerNorm = 255 * markers
        outPredMarkerUint8 = outPredMarkerNorm.astype(np.uint8)
        
        fname = outname.replace('tif','png')
        fname = fname.replace('t','mask')
         
        imageio.imwrite(maskDir + fname, outPredMaskUint8)
        imageio.imwrite(markerDir + fname, outPredMarkerUint8) 

      
    finalTime = time.time() - startTime
    msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(finalTime))
    print(msg)  

if __name__ == '__main__':
    main()

# Train UNet, USENet, UMiTNet, MUNet, MUSENet, and MUMiTNet models and extract masks and markers for trained / pre-trained models. 

There are three parts for this software in ```src``` folder, you can skip Part 1 (Train Models) if you are planning to use pre-trained models.

**Part 1 -->** Generate Masks and Markers from Silver Truth (ST): generating binary masks and markers to train the network models. 

**Part 2 -->** Train Models: train all models from scratch.

**Part 3 -->** Extract Masks and Markers: use trained/pre-trained models to extract masks and markers.

**Part 4 -->** Post-Processing: use post-processing to get labeled results from masks and markers.

In every parts, there are readme file that describes the needed steps. The description is also placed here.

**You need to use PyTorch to do Part 2 and Part 3.**

**You need to use MATLAB to do Part 1 and Part 4. However, python implementation for post-processing is also provided, but not as accurate as MATLAB one. Part 1 can also be integrated using python implementation as well.**

## Part 1 : Generate Masks and Markers from Silver Truth (ST)

The MATLAB codes used to generate binary masks and markers from ST is given in ```dataset``` folder.

1. Specify dataset path along with the parameters to remove small blobs and fill blobs will small holes in ```runGetMarker.m``` script and run it. 

2. The ```getMarker.m``` script will generate binary masks by binarizing cells in ```ST``` image and save it in ```BSEG``` folder. Moreover, it will generate binary markers by eroding (erosion parameter is selected according to the cell area) cells in ```ST``` image and save it in ```MARKER``` folder. 


## Part 2 : Train Models

**To train UNet, USENet, UMiTNet**

1. Put your data used to train the network in a folder called ```dataset/train/``` folder. DIC-C2DH-HeLa is given as an example. Please cite Cell Tracking Challenge papers if you use DIC-C2DH-HeLa in your work.

* From the provided ST (silver truth) from ```SEG``` folder, binary segmentations (```BSEG```) were obtained by binarizing the cells in the ST and by applying erosion to each individual cells in the ST markers (```MARKER```) were obtained.   

2. Run ```run_train_UNet.sh``` for UNet and other .sh files for other networks. 

This script will train UNet models according to the data you provided and save trained model inside ```src/models``` folder.

**To train MUNet, MUSENet, MUMiTNet**

1. Put your data used to train the network in a folder called ```data/train/``` folder. DIC-C2DH-HeLa is given as an example. Please cite Cell Tracking Challenge papers if you use DIC-C2DH-HeLa in your work.

2.  Background Subtraction (BGS) and flux masks are optained using traditional methods. DIC-C2DH-HeLa is given as an example.

3. Run ```run_train_MUNet.sh``` for MUNet and other .sh files for other networks. 

This script will train MUNet models according to the data you provided and save trained model inside ```src/models``` folder.


## Part 3 : Extract Masks and Markers

**To extract masks and markers of UNet, USENet, UMiTNet**

1. To extract masks and markers using trained / pre-trained model of UNet, USENet, UMiTNet create a new folder with dataset name inside ```dataset/test/``` folder and and put your data inside created folder.

2. Change dataset paths accordingly in ```inferUNet.py``` or other inference scripts you want to use.

3. Change video sequence paths accordingly in ```files/seqNum.txt```. Some examples of video sequence taken from cell tracking challenge are given inside  ```seqNum.txt```

4. Run ```run_infer_UNet.sh``` for UNet and other .sh files for other networks.

This script will extract masks using trained / pre-trained model of UNet for the given dataset and save the result of output masks and markers inside ```output``` folder.

**To extract masks and markers of MUNet, MUSENet, MUMiTNet**

1. To extract masks and markers using trained / pre-trained model of MUNet, MUSENet, MUMiTNet:

* * create a new folder with dataset name inside ```data/test/``` folder and and put your data inside created folder. 

* * create another folder inside ```data/test/``` folder and put Background Subtraction (BGS) masks related to the data. Background subtraction masks are given as an example for DIC-C2DH-HeLa in ```data/train/``` folder, which is obtained using OpenCV library **BackgroundSubtractorMOG2** on an input images. 

* * create another folder inside ```data/test/``` folder and put Flux masks related to the data. Flux masks are given as an example DIC-C2DH-HeLa in ```data/train/``` folder, which is obtained using **trace of the flux tensor** on an input images.

* * For more detail how to obtain Background Subtraction and Flux Tensor read the paper.  

2. Change dataset paths accordingly in ```inferMUNet.py``` or other inference scripts you want to use.

3. Change video sequence paths accordingly in ```files/seqNum.txt```. Some examples of video sequence taken from cell tracking challenge are given inside  ```seqNum.txt```

4. Run ```run_infer_MUNet.sh``` for MUNet and other .sh files for other networks.

This script will extract masks using trained / pre-trained model of MUNet for the given dataset with related background subtraction and flux masks and save the result of output masks and markers inside ```output``` folder.
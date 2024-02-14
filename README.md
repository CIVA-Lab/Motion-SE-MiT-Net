# Motion-SE-MiT-Net
AIPR2023 - Marker and Motion Guided Deep Networks for Cell Segmentation and Detection Using Weakly Supervised Microscopy Data
<!-- The official implementation of the ICPR 2020 paper [**Motion U-Net: Multi-cue Encoder-Decoder Network for Motion Segmentation**](https://ieeexplore.ieee.org/document/9413211) -->

 The slides are available [Marker and Motion Guided Deep Networks for Cell Segmentation and Detection Using Weakly Supervised Microscopy Data](/figures/Slides.pdf)

 </br>

## News

**[February 9, 2024]**

- :fire::fire::fire:  **Code for generating Background Subtraction (BGS) result using OpenCV library used in this work is available now !** 

**[February 8, 2024]** 

Post-processing MATLAB and BGS codes will be uploaded soon

- :fire::fire::fire:  **BGS algorithm and how to use it will be uploaded soon.** 

- :fire::fire::fire:  **Post-processing MATLAB scripts for getting final labeled result from masks and markers is uploaded.** 

- :fire::fire::fire:  **Weights and instructions of how to train and inference the models are uploaded.** 

- :fire::fire::fire:  **MATLAB scripts for getting binary masks and markers from Silver Truth (ST) is uploaded.** 


**[February 7, 2024]** 

- :fire::fire::fire:  **The src codes are uploaded. Instructions of how to use them and the weights will be uploaded soon** 

</br>

## Marker and Motion Guided Deep Networks for Cell Segmentation and Detection Using Weakly Supervised Microscopy Data
The accurate detection and segmentation of cells in microscopy image sequences play a crucial role in biomedical research and clinical diagnostic applications. However, accurately segmenting cells in low signal-to-noise ratio images remains challenging due to dense touching cells and deforming cells with indistinct boundaries. To address these challenges, this paper investigates the effectiveness of marker-guided networks, including UNet, with Squeeze-and-Excitation (SE) or MixTransformer (MiT) backbone architectures. We explore their performance both independently and in conjunction with motion cues, aiming to enhance cell segmentation and detection in both real and simulated data. The squeeze and excitation blocks enable the network to recalibrate features, highlighting valuable ones while downplaying less relevant ones. In contrast, the transformer encoder doesn’t require positional encoding, eliminating the need for interpolating positional codes, which can result in reduced performance when the testing resolution differs from the training data. We propose novel deep architectures, namely Motion USENet (MUSENet) and Motion UMiTNet (MUMiTNet), and adopt our previous method Motion UNet (MUNet), for robust cell segmentation and detection. Motion and change cues are computed through our tensor-based motion estimation and multimodal background subtraction (BGS) modules. The proposed network was trained, tested, and evaluated on the Cell Tracking Challenge (CTC) dataset. When comparing UMiTNet to USENet, there is a noteworthy 23% enhancement in cell segmentation and detection accuracy when trained on real data and tested on simulated data. Additionally, there is a substantial 32% improvement when trained on simulated data and tested on real data. Introducing motion cues (MUMiTNet) resulted in a significant 25% accuracy improvement over UMiTNet when trained on real data and tested on simulated data, and a 9% improvement when trained on simulated data and tested on real data.

![](/figures/initialFigure.PNG)


## Pipeline and Network Architecture
This is a comprehensive overview of the proposed pipeline using three different encoders. To start, the pipeline commences with an initial preprocessing phase, aimed to enhance the visibility of cells in the input images. Subsequently, the preprocessed data is expanded through augmentation and then input into the network. Moreover, motion cues are computed and fed into the network as a separate stream. Following this, the network’s output masks undergo post-processing utilizing mathematical morphology. Finally, the watershed technique is implemented to distinguish interconnected cells from one another, where a labeled mask would be the final output from a pipeline in which each cell is assigned a distinct identifier.

![](/figures/Overall-Pipeline.png)


## Qualitative Results
The qualitative results of proposed deep learning models with motion cue integration, MUNet, MUSENet, and MUMiTNet are shown below.

![](/figures/QualitativeFigure.png)

</br>

# Pre-trained weights of Motion-SE-MiT-Net
If you want to use pre-trained weights, put them inside **src/models/** folder.

```Combine-All-2D-USENet.pt``` is the single model (USENet) trained using 6 different data from cell tracking challenge and has higher accuracy than UNet or UMiTNet on cell tracking challenge data. 

Link to download [**USENet weights**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Combine-All-2D-USENet.pt)

```Combine-All-2D-MUMiTNet.pt``` is the single model (MUMiTNet) trained using 6 different data from cell tracking challenge and has a close accuracy as USENet on cell tracking challenge data. 

Link to download [**MUMiTNet weights**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Combine-All-2D-MUMiTNet.pt)

```Fluo-N2DH-GOWT1-MUMiTNet.pt``` is the model trained (MUMiTNet) trained using only **Fluo-N2DH-GOWT1 (real data)** from cell tracking challenge dataset and used to test on unseen **Fluo-N2DH-SIM+ (synthetic data)**.

Link to download [**MUMiTNet weights trained on Fluo-N2DH-GOWT1 (real data)**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Fluo-N2DH-GOWT1-MUMiTNet.pt)

```Fluo-N2DH-SIM+-MUMiTNet.pt``` is the model trained (MUMiTNet) trained using only **Fluo-N2DH-SIM+ (synthetic data)** from cell tracking challenge dataset and used to test on unseen **Fluo-N2DH-GOWT1 (real data)**.

Link to download [**MUMiTNet weights trained on Fluo-N2DH-SIM+ (synthetic data)**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Fluo-N2DH-SIM+-MUMiTNet.pt)

</br>

# How to use Motion-SE-MiT-Net

```src``` folder contains all scripts used to train models, extract masks from trained models, and post-processing the output results to get labeled masks.

```weights``` folder contains pre-trained weights of the Motion-SE-MiT-Net, if you want to use pre-trained weights, put them inside ```src/models``` folder.


There are three parts for this software in ```src``` folder, you can skip Part 1 (Train Models) if you are planning to use pre-trained models.

**Part 1 -->** Generate Masks and Markers from Silver Truth (ST): generating binary masks and markers to train the network models. 

**Part 2 -->** Train Models: train all models from scratch.

**Part 3 -->** Extract Masks and Markers: use trained/pre-trained models to extract masks and markers.

**Part 4 -->** Post-Processing: use post-processing to get labeled results from masks and markers.

**Part 5 -->** Evaluation: evaluate the final labeled result using evaluation methodology provided by Cell Tracking Challenge.

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

## Part 4 : Post-Processing

The MATLAB codes used to get labeled result from masks and markers is given in ```src``` folder.

1. Specify output path along with the parameters to remove small blobs and fill blobs will small holes and threshold values for converting masks and markers into binary masks and markers in ```runPostProcessing.m``` script and run it. 

2. The ```postProcessing.m``` script will generate final labeled result using masks and markers from network output and applying **watershed algorithm** and save the result in ```LabelMat``` folder. Moreover, the script will also generate the general visualization images showing mask, marker, overlay of marker on mask, and final labeled result using different color for each cell and save it in ```LabelMat_VIS``` folder.


## Part 5: Evaluation

The evaluation script provided in Cell Tracking Challenge is used. The details and the script can be found here [Evaluation Methodology](http://celltrackingchallenge.net/evaluation-methodology/). 

</br>

## Running OpenCV Background Subtraction (BGS):

**To get BGS results for use in MUNet, MUSENet, MUMiTNet**

1. Go to ```OpenCV_BGS``` folder.
```
cd OpenCV_BGS
```

2. Change the input/output paths and image file format in ```config.txt``` file accordingly.  

3. Create a ```build``` folder:  
```
mkdir build
```

4. Enter the ```build``` folder:
```
cd build
```

5. Run ```cmake```:
```
cmake ..
```

6. Run ```make```:
```
make
```

7. Go to ```bin/linux``` folder:
```
cd ../bin/linux
```

8. Run ```BGSubOpenCV```:
```
./BGSubOpenCV
```

</br>

## Project Collaborators and Contact

**Author:** Gani Rahmon, and Kannappan Palaniappan

Copyright &copy; 2023-2024. Gani Rahmon and Prof. K. Palaniappan and Curators of the University of Missouri, a public corporation. All Rights Reserved.

**Created by:** Ph.D. student: Gani Rahmon  
Department of Electrical Engineering and Computer Science,  
University of Missouri-Columbia  

For more information, contact:

* **Gani Rahmon**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
grzc7@mail.missouri.edu  

* **Dr. K. Palaniappan**  
205 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
palaniappank@missouri.edu

</br>

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{gani2021MUNet,
  title={Motion U-Net: Multi-cue Encoder-Decoder Network for Motion Segmentation}, 
  author={Rahmon, Gani and Bunyak, Filiz and Seetharaman, Guna and Palaniappan, Kannappan},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, 
  pages={8125-8132},
  year={2021}
}

@Article{Rahmon2023DeepFTSG,
    author={Rahmon, Gani and Palaniappan, Kannappan and Toubal, Imad Eddine and Bunyak, Filiz and Rao, Raghuveer and Seetharaman, Guna},
    title={DeepFTSG: Multi-stream Asymmetric USE-Net Trellis Encoders with Shared Decoder Feature Fusion Architecture for Video Motion Segmentation},
    journal={International Journal of Computer Vision},
    year={2023},
    month={Oct},
    day={17},
    issn={1573-1405}
}
```

## ✏️ Citation

Site cell tracking challenge (CTC) papers if you use CTC data in your project:

```
@ARTICLE{maskaCTC,
  author={M. Maška and V. Ulman and P. Delgado-Rodriguez and E. Gómez-de-Mariscal and T. Nečasová and F. A. Guerrero Peña and T. I. Ren and etc.},
  journal={Nature Methods}, 
  title="{The Cell Tracking Challenge: 10 years of objective benchmarking}", 
  year={2023},
  volume={20},
  number={},
  pages={1010-1020}
}

@ARTICLE{ulmanCTC,
  author={V. Ulman and M. Maška  and K. Magnusson and O. Ronneberger and C. Haubold and N. Harder and P. Matula and P. Matula and etc.},
  journal={Nature Methods}, 
  title="{An objective comparison of cell-tracking algorithms}", 
  year={2017},
  volume={14},
  number={},
  pages={1141-1152}
}
```

## ✏️ Citation

Site OpenCV background subraction papers if you use OpenCV_BGS code in your project:

```
@article{Zivkovic,
    title       = "{Efficient adaptive density estimation per image pixel for the task of background subtraction}",
    journal     = "Pattern Recognition Letters",
    volume      = "27",
    number      = "7",
    pages       = "773 - 780",
    year        = "2006",
    author      = "Zivkovic, Z. and van der Heijden, F."
}

@inproceedings{Zivkovic2,
  author        = "Zivkovic, Z.",
  booktitle     = "Int. Conf. Pattern Recognition", 
  title         = "{Improved adaptive Gaussian mixture model for background subtraction}", 
  year          = "2004",
  volume        = "2",
  pages         = "28-31"
}
```


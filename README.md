# Motion-SE-MiT-Net
AIPR2023 - Marker and Motion Guided Deep Networks for Cell Segmentation and Detection Using Weakly Supervised Microscopy Data
<!-- The official implementation of the ICPR 2020 paper [**Motion U-Net: Multi-cue Encoder-Decoder Network for Motion Segmentation**](https://ieeexplore.ieee.org/document/9413211) -->

## News

**[November 30, 2023]** 

- :fire::fire::fire:  **The src code and weight will be uploaded soon** 


## Marker and Motion Guided Deep Networks for Cell Segmentation and Detection Using Weakly Supervised Microscopy Data
The accurate detection and segmentation of cells in microscopy image sequences play a crucial role in biomedical research and clinical diagnostic applications. However, accurately segmenting cells in low signal-to-noise ratio images remains challenging due to dense touching cells and deforming cells with indistinct boundaries. To address these challenges, this paper investigates the effectiveness of marker-guided networks, including UNet, with Squeeze-and-Excitation (SE) or MixTransformer (MiT) backbone architectures. We explore their performance both independently and in conjunction with motion cues, aiming to enhance cell segmentation and detection in both real and simulated data. The squeeze and excitation blocks enable the network to recalibrate features, highlighting valuable ones while downplaying less relevant ones. In contrast, the transformer encoder doesn’t require positional encoding, eliminating the need for interpolating positional codes, which can result in reduced performance when the testing resolution differs from the training data. We propose novel deep architectures, namely Motion USENet (MUSENet) and Motion UMiTNet (MUMiTNet), and adopt our previous method Motion UNet (MUNet), for robust cell segmentation and detection. Motion and change cues are computed through our tensor-based motion estimation and multimodal background subtraction (BGS) modules. The proposed network was trained, tested, and evaluated on the Cell Tracking Challenge (CTC) dataset. When comparing UMiTNet to USENet, there is a noteworthy 23% enhancement in cell segmentation and detection accuracy when trained on real data and tested on simulated data. Additionally, there is a substantial 32% improvement when trained on simulated data and tested on real data. Introducing motion cues (MUMiTNet) resulted in a significant 25% accuracy improvement over UMiTNet when trained on real data and tested on simulated data, and a 9% improvement when trained on simulated data and tested on real data.

![](/figures/initialFigure.PNG)


## Pipeline and Network Architecture
This is a comprehensive overview of the proposed pipeline using three different encoders. To start, the pipeline commences with an initial preprocessing phase, aimed to enhance the visibility of cells in the input images. Subsequently, the preprocessed data is expanded through augmentation and then input into the network. Moreover, motion cues are computed and fed into the network as a separate stream. Following this, the network’s output masks undergo post-processing utilizing mathematical morphology. Finally, the watershed technique is implemented to distinguish interconnected cells from one another, where a labeled mask would be the final output from a pipeline in which each cell is assigned a distinct identifier.

![](/figures/Overall-Pipeline.png)


## Qualitative Results
The qualitative results of proposed deep learning models with motion cue integration, MUNet, MUSENet, and MUMiTNet are shown below.

![](/figures/QualitativeFigure.png)


# How to use Motion-SE-MiT-Net

**src** folder contains all scripts used to train models, extract masks from trained models, and post-processing the output results to get labeled masks.

**weights** folder contains pre-trained weights of the Motion-SE-MiT-Net, if you want to use pre-trained weights, put them inside **src/weights/** folder.


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
```
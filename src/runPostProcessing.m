clear all; close all;

% DIC
postProcessing("output/train/UNet/", "DIC-C2DH-HeLa/", "01/", 0.3, 0.5, 300);
postProcessing("output/train/UNet/", "DIC-C2DH-HeLa/", "02/", 0.3, 0.5, 300);

% % Fluo-HeLa
% postProcessing("output/train/UNet/", "Fluo-N2DL-HeLa/", "01/", 0.9, 0.9, 10);
% postProcessing("output/train/UNet/", "Fluo-N2DL-HeLa/", "02/", 0.9, 0.9, 10);
% 
% % PhC
% postProcessing("output/train/UNet/", "PhC-C2DH-U373/", "01/", 0.9, 0.9, 100);
% postProcessing("output/train/UNet/", "PhC-C2DH-U373/", "02/", 0.9, 0.9, 100);
% 
% % MSC
% postProcessing("output/train/UNet/", "Fluo-C2DL-MSC/", "01/", 0.8, 0.8, 200);
% postProcessing("output/train/UNet/", "Fluo-C2DL-MSC/", "02/", 0.8, 0.8, 200);
% 
% % GOWT1
% postProcessing("output/train/UNet/", "Fluo-N2DH-GOWT1/", "01/", 0.9, 0.9, 10);
% postProcessing("output/train/UNet/", "Fluo-N2DH-GOWT1/", "02/", 0.9, 0.9, 10);
% 
% % PSC
% postProcessing("output/train/UNet/", "PhC-C2DL-PSC/", "01/", 0.9, 0.9, 10);
% postProcessing("output/train/UNet/", "PhC-C2DL-PSC/", "02/", 0.9, 0.9, 10);
% 
% % Huh7
% postProcessing("output/train/UNet/", "Fluo-C2DL-Huh7/", "01/", 0.9, 0.9, 100);
% postProcessing("output/train/UNet/", "Fluo-C2DL-Huh7/", "02/", 0.9, 0.9, 100);
% 
% % HSC
% postProcessing("output/train/UNet/", "BF-C2DL-HSC/", "01/", 0.9, 0.9, 10);
% postProcessing("output/train/UNet/", "BF-C2DL-HSC/", "02/", 0.9, 0.9, 10);
% 
% % MuSC
% postProcessing("output/train/UNet/", "BF-C2DL-MuSC/", "01/", 0.9, 0.9, 100);
% postProcessing("output/train/UNet/", "BF-C2DL-MuSC/", "02/", 0.9, 0.9, 100);
% 
% % SIM+
% postProcessing("output/train/UNet/", "Fluo-N2DH-SIM+/", "01/", 0.9, 0.9, 10);
% postProcessing("output/train/UNet/", "Fluo-N2DH-SIM+/", "02/", 0.9, 0.9, 10);

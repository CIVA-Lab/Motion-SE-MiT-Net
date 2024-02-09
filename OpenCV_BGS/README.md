# Running OpenCV Background Subtraction (BGS):

**To get BGS results for use in MUNet, MUSENet, MUMiTNet**

1. Change the input/output paths and image file format in ```config.txt``` file accordingly. 
```
# Config file to run OpenCV Background Subtraction

##### IO Parameters #####
# Input sequence path (give full path)
input_dir = /mnt/d/GitHub/Motion-SE-MiT-Net-AIPR2023/src/dataset/train/DIC-C2DH-HeLa/01 

# Input image file format (e.g., jpg, png)
image_ext = tif

# Ouput path (give full path)
output_dir =  /mnt/d/GitHub/Motion-SE-MiT-Net-AIPR2023/src/dataset/train/DIC-C2DH-HeLa/01_BGS/ 
``` 

2. Create a ```build``` folder:  
```
mkdir build
```

3. Enter the ```build``` folder:
```
cd build
```

4. Run ```cmake```:
```
cmake ..
```

5. Run ```make```:
```
make
```

6. Go to ```bin/linux``` folder:
```
cd ../bin/linux
```

7. Run ```BGSubOpenCV```:
```
./BGSubOpenCV
```

8. The output of BGS will be saved in the provided output path.
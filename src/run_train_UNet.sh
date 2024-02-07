python -u trainUNet.py --dataset DIC-C2DH-HeLa
python -u trainUNet.py --dataset Fluo-N2DL-HeLa
python -u trainUNet.py --dataset Fluo-N2DH-GOWT1
python -u trainUNet.py --dataset PhC-C2DH-U373
python -u trainUNet.py --dataset Fluo-C2DL-MSC --resize_to 832 992
python -u trainUNet.py --dataset PhC-C2DL-PSC
python -u trainUNet.py --dataset BF-C2DL-HSC
python -u trainUNet.py --dataset BF-C2DL-MuSC

python -u trainUNetSim.py --dataset Fluo-N2DH-SIM+ --resize_to 690 628

python -u trainAllUNet.py --dataset Combine-All-2D --resize_to 512 512 --train_resolution 256 256

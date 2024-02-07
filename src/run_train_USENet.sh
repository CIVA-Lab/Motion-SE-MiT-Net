python -u trainUSENet.py --dataset DIC-C2DH-HeLa
python -u trainUSENet.py --dataset PhC-C2DH-U373
python -u trainUSENet.py --dataset Fluo-N2DL-HeLa
python -u trainUSENet.py --dataset Fluo-C2DL-MSC --resize_to 832 992
python -u trainUSENet.py --dataset Fluo-N2DH-GOWT1
python -u trainUSENet.py --dataset PhC-C2DL-PSC
python -u trainUSENet.py --dataset BF-C2DL-HSCC  
python -u trainUSENet.py --dataset BF-C2DL-MuSC

python -u trainUSENetSim.py --dataset Fluo-N2DH-SIM+ --resize_to 690 628

python -u trainAllUSENet.py --dataset Combine-All-2D --resize_to 512 512 --train_resolution 256 256


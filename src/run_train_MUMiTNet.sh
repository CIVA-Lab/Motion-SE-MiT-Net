python -u trainMUMiTNet.py --dataset DIC-C2DH-HeLa
python -u trainMUMiTNet.py --dataset Fluo-N2DL-HeLa
python -u trainMUMiTNet.py --dataset Fluo-N2DH-GOWT1
python -u trainMUMiTNet.py --dataset PhC-C2DH-U373
python -u trainMUMiTNet.py --dataset Fluo-C2DL-MSC --resize_to 832 992
python -u trainMUMiTNet.py --dataset PhC-C2DL-PSC
python -u trainMUMiTNet.py --dataset BF-C2DL-HSC
python -u trainMUMiTNet.py --dataset BF-C2DL-MuSC

python -u trainMUMiTNetSim.py --dataset Fluo-N2DH-SIM+ --resize_to 690 628

python -u trainAllMUMiTNet.py --dataset Combine-All-2D --resize_to 512 512 --train_resolution 256 256


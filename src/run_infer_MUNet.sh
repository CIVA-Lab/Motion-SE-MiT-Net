# Infer on Train Data 
python -u inferMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset PhC-C2DL-PSC  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset PhC-C2DL-PSC  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset BF-C2DL-HSC  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset BF-C2DL-HSC  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset BF-C2DL-MuSC  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset BF-C2DL-MuSC  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 02  --datatype train

python -u inferMUNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 01  --datatype train
python -u inferMUNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 02  --datatype train


# Infer on Train Data using Single Trained Model
python -u inferAllMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype train

python -u inferAllMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype train

python -u inferAllMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype train

python -u inferAllMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype train

python -u inferAllMUNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype train

python -u inferAllMUNet.py --dataset PhC-C2DL-PSC  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset PhC-C2DL-PSC  --sequence_id 02  --datatype train

python -u inferAllMUNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 02  --datatype train

python -u inferAllMUNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 01  --datatype train
python -u inferAllMUNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 02  --datatype train


# Infer on Test Data using Single Trained Model
python inferMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype test
python inferMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype test

python inferMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype test
python inferMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype test

python inferMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype test
python inferMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype test

python inferMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype test
python inferMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype test

python inferMUNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype test
python inferMUNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype test


# Infer on Test Data using Single Trained Model
python inferAllMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype test
python inferAllMUNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype test

python inferAllMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype test
python inferAllMUNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype test

python inferAllMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype test
python inferAllMUNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype test

python inferAllMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype test
python inferAllMUNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype test

python inferAllMUNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype test
python inferAllMUNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype test


# Infer on Simulated and Original Data
python -u inferMUNetSimOrg.py --dataset Fluo-N2DH-SIM+ --model Fluo-N2DH-GOWT1 --sequence_id 01  --datatype train
python -u inferMUNetSimOrg.py --dataset Fluo-N2DH-SIM+ --model Fluo-N2DH-GOWT1 --sequence_id 02  --datatype train

python -u inferMUNetSimOrg.py --dataset Fluo-N2DH-GOWT1 --model Fluo-N2DH-SIM+ --sequence_id 01  --datatype train
python -u inferMUNetSimOrg.py --dataset Fluo-N2DH-GOWT1 --model Fluo-N2DH-SIM+ --sequence_id 02  --datatype train
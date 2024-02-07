# Infer on Train Data 
python -u inferUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset PhC-C2DL-PSC  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset PhC-C2DL-PSC  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset BF-C2DL-HSC  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset BF-C2DL-HSC  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset BF-C2DL-MuSC  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset BF-C2DL-MuSC  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset Fluo-C2DL-Huh7  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset Fluo-C2DL-Huh7  --sequence_id 02  --datatype train

python -u inferUSENet.py --dataset Fluo-N2DH-SIM+  --sequence_id 01  --datatype train
python -u inferUSENet.py --dataset Fluo-N2DH-SIM+  --sequence_id 02  --datatype train


# Infer on Train Data using Single Trained Model
python -u inferAllUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype train

python -u inferAllUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype train

python -u inferAllUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype train

python -u inferAllUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype train

python -u inferAllUSENet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype train

python -u inferAllUSENet.py --dataset PhC-C2DL-PSC  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset PhC-C2DL-PSC  --sequence_id 02  --datatype train

python -u inferAllUSENet.py --dataset Fluo-C2DL-Huh7  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset Fluo-C2DL-Huh7  --sequence_id 02  --datatype train

python -u inferAllUSENet.py --dataset Fluo-N2DH-SIM+  --sequence_id 01  --datatype train
python -u inferAllUSENet.py --dataset Fluo-N2DH-SIM+  --sequence_id 02  --datatype train


# Infer on Test Data using Single Trained Model
python inferUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype test
python inferUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype test

python inferUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype test
python inferUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype test

python inferUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype test
python inferUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype test

python inferUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype test
python inferUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype test

python inferUSENet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype test
python inferUSENet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype test


# Infer on Test Data using Single Trained Model
python inferAllUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype test
python inferAllUSENet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype test

python inferAllUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype test
python inferAllUSENet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype test

python inferAllUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype test
python inferAllUSENet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype test

python inferAllUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype test
python inferAllUSENet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype test

python inferAllUSENet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype test
python inferAllUSENet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype test


# Infer on Simulated and Original Data
python -u inferUSENetSimOrg.py --dataset Fluo-N2DH-SIM+ --model Fluo-N2DH-GOWT1 --sequence_id 01  --datatype train
python -u inferUSENetSimOrg.py --dataset Fluo-N2DH-SIM+ --model Fluo-N2DH-GOWT1 --sequence_id 02  --datatype train

python -u inferUSENetSimOrg.py --dataset Fluo-N2DH-GOWT1 --model Fluo-N2DH-SIM+ --sequence_id 01  --datatype train
python -u inferUSENetSimOrg.py --dataset Fluo-N2DH-GOWT1 --model Fluo-N2DH-SIM+ --sequence_id 02  --datatype train
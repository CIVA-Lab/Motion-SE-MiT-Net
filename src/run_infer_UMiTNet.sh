# Infer on Train Data 
python -u inferUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset PhC-C2DL-PSC  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset PhC-C2DL-PSC  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset BF-C2DL-HSC  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset BF-C2DL-HSC  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset BF-C2DL-MuSC  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset BF-C2DL-MuSC  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 02  --datatype train

python -u inferUMiTNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 01  --datatype train
python -u inferUMiTNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 02  --datatype train


# Infer on Train Data using Single Trained Model
python -u inferAllUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype train

python -u inferAllUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype train

python -u inferAllUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype train

python -u inferAllUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype train

python -u inferAllUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype train

python -u inferAllUMiTNet.py --dataset PhC-C2DL-PSC  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset PhC-C2DL-PSC  --sequence_id 02  --datatype train

python -u inferAllUMiTNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset Fluo-C2DL-Huh7  --sequence_id 02  --datatype train

python -u inferAllUMiTNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 01  --datatype train
python -u inferAllUMiTNet.py --dataset Fluo-N2DH-SIM+  --sequence_id 02  --datatype train


# Infer on Test Data using Single Trained Model
python inferUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype test
python inferUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype test

python inferUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype test
python inferUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype test

python inferUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype test
python inferUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype test

python inferUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype test
python inferUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype test

python inferUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype test
python inferUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype test


# Infer on Test Data using Single Trained Model
python inferAllUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 01  --datatype test
python inferAllUMiTNet.py --dataset DIC-C2DH-HeLa  --sequence_id 02  --datatype test

python inferAllUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 01  --datatype test
python inferAllUMiTNet.py --dataset Fluo-N2DL-HeLa  --sequence_id 02  --datatype test

python inferAllUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 01  --datatype test
python inferAllUMiTNet.py --dataset Fluo-C2DL-MSC  --sequence_id 02  --datatype test

python inferAllUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 01  --datatype test
python inferAllUMiTNet.py --dataset Fluo-N2DH-GOWT1  --sequence_id 02  --datatype test

python inferAllUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 01  --datatype test
python inferAllUMiTNet.py --dataset PhC-C2DH-U373  --sequence_id 02  --datatype test


# Infer on Simulated and Original Data
python -u inferUMiTNetSimOrg.py --dataset Fluo-N2DH-SIM+ --model Fluo-N2DH-GOWT1 --sequence_id 01  --datatype train
python -u inferUMiTNetSimOrg.py --dataset Fluo-N2DH-SIM+ --model Fluo-N2DH-GOWT1 --sequence_id 02  --datatype train

python -u inferUMiTNetSimOrg.py --dataset Fluo-N2DH-GOWT1 --model Fluo-N2DH-SIM+ --sequence_id 01  --datatype train
python -u inferUMiTNetSimOrg.py --dataset Fluo-N2DH-GOWT1 --model Fluo-N2DH-SIM+ --sequence_id 02  --datatype train
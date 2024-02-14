# Pre-trained weights of Motion-SE-MiT-Net
This folder contains pre-trained wights of Motion-SE-MiT-Net. If you want to use pre-trained weights, put them inside **src/models/** folder.

```Combine-All-2D-USENet.pt``` is the single model (USENet) trained using 6 different data from cell tracking challenge and has higher accuracy than UNet or UMiTNet on cell tracking challenge data. 

Link to download [**USENet weights**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Combine-All-2D-USENet.pt)

```Combine-All-2D-MUMiTNet.pt``` is the single model (MUMiTNet) trained using 6 different data from cell tracking challenge and has a close accuracy as USENet on cell tracking challenge data. 

Link to download [**MUMiTNet weights**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Combine-All-2D-MUMiTNet.pt)

```Fluo-N2DH-GOWT1-MUMiTNet.pt``` is the model trained (MUMiTNet) trained using only **Fluo-N2DH-GOWT1 (real data)** from cell tracking challenge dataset and used to test on unseen **Fluo-N2DH-SIM+ (synthetic data)**.

Link to download [**MUMiTNet weights trained on Fluo-N2DH-GOWT1 (real data)**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Fluo-N2DH-GOWT1-MUMiTNet.pt)

```Fluo-N2DH-SIM+-MUMiTNet.pt``` is the model trained (MUMiTNet) trained using only **Fluo-N2DH-SIM+ (synthetic data)** from cell tracking challenge dataset and used to test on unseen **Fluo-N2DH-GOWT1 (real data)**.

Link to download [**MUMiTNet weights trained on Fluo-N2DH-SIM+ (synthetic data)**](https://meru.rnet.missouri.edu/~grzc7/Motion_SE_MiT_Net_AIPR2023_weights/Fluo-N2DH-SIM+-MUMiTNet.pt)
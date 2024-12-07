# Few-shot Domain Adaptation for Cross-Modality Image Segmentation

Authors: Skyler Grandel and Dung (Judy) Nguyen

-----
For this project we investigated domain adaptation for medical image segmentation on the subject of tumorous brain scans, transferring domains from Magnetic Resonance Images (MRIs) to to Computed Tomography (CT) scans. MRIs are know to be superior to CT scans for identifying and segmenting brain
tumors, but CT scans are faster and cheaper.
> How a segmentation model trained on MRI can be adapted on CT dataset?

**Key contribution:**
- Study performance medical segmentation model on unseen domain (source domain: MRI, target domain: CT scans).
- Domain adaptation to adapt MRI-trained model to CT scans.
- Study effect of data augmentation within the context of domain adaptation problem.

## How to run
- Install neccessary packages:
  ```
  pip3 install --no-cache-dir -r requirements.txt
  ```
- We used two datasets:
- Datasets: 
  - MRI: [kaggle-link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
  - CT: https://www.kaggle.com/datasets/mahmoudshaheen1134/brain-tumor-dataset/data
  - This step can be done via by downloading `.zip` or using `kaggle-hub`
- Pre-trained model ckpt: https://vanderbilt.box.com/s/kcb6n5n3e92janqnbvhmrsffsyardkzq

- Interactive notebook files for running inferences on both datasets: [test_brain_tumors.ipynb](notebooks/test_brain_tumors.ipynb)

- Data augmentation: [data_augmentation.ipynb](notebooks/data_augmentation.ipynb)

- Train a U-net model from scratch: 
  ```
  python train.py
  ```

- To start fine-tuning:
  ```
  python3 finetune_no_augmentation.py --epochs 10
  ```

  ```
  python3 finetune_brain_tumors.py --batch_size 1 --lr 0.001 --save_model no_aug_model.pth --epochs 10
  ```

## Breakdown of the code structure

```
├── notebooks                   <- contains interactive python files
│   ├── test_brain_tumors.ipynb       <- testing on both datasets
│   ├── data_augmentation.ipynb       <- ITK augmentation
├── datasets                    <- downloaded datasets
├── helpers                     <- utilities
│   ├── plot_helper             
│   ├── augmentation_helper     
├── weights                     <- saved models and plots
├── datasets.py                 <- loading datasets, augmentation
├── finetune_brain_tumors.py    <- finetuning with augmentation
├── finetune_no_augmentation.py <- finetuning no augmentation
├── train.py                    <- training model wit MRI data
├── .gitignore                <- List of files ignored by git
```

The graph below lists out our key main functions with modification compared to original works we acknowledged.

## Acknowledgement
- https://github.com/mateuszbuda/brain-segmentation-pytorch
- https://www.kaggle.com/code/abdallahwagih/brain-tumor-segmentation-unet-efficientnetb7
- Buda, Mateusz, Ashirbani Saha, and Maciej A. Mazurowski. "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in biology and medicine 109 (2019): 218-225.
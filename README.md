# Self-supervised Approach for Retinal OCT Scan Classification using SimCLR

![alt text](https://mbzuai.ac.ae/application/themes/mbzuai/dist/images/mbzuai_logo.png)

Authors: Diego Saenz, Faris AlMalik, Abdulwahab Sahyoun 



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdu355/ml701_project_grp20/blob/main/ml701_Proj_Final.ipynb)

## Imports & Initialization
1. Install packages with pip: 
  * `pip install pytorch-lightning image comet-ml captum flask_compress Pillow`
  * `pip install git+https://github.com/PytorchLightning/pytorch-lightning-bolts.git@master --upgrade`
2. Import all libraries from "Importing libraries and Dataset" section in the notebook.

## Finetuning & Evaluation
1. Download the following datasets & add them to your Google Drive for loading:
  - Mendeley training and testing dataset (Kermany et al. [8]): URL
  - Duke University testing dataset (Srinivasan et al. [14]) + Github testing dataset (Wintergerst et al. [16]): URL
2. Initialize `OCTDataModule` & Image Transformer `transforms.Compose`
3. Check a sample of the dataset using 
  * `ds_train = oct_data.oct_train` 
  * `tvf.to_pil_image(ds_train[6][0][0])`
4. Initialize the `OCTModel`
  * Encoder/Base Model: Transfer weights from simCLR to the base encoder ` self.base_model = SimCLR.load_from_checkpoint(self.hparams.embeddings_path, strict=False)`   
  * For the optimizer use: `Adagrad, SGD, Adam, LBFGS, RMSProp, Adamax` 
  * LR Scheduler: `OneCycleLR`
5. Define Hyperparams:
`hparams = Namespace(
    learning_rate=1e-3,
    freeze_base=True,
    tune=True,
    max_epochs=15,
    steps_per_epoch = len(oct_data.train_dataloader()),
    n_classes=num_of_classes,
    embeddings_path=weight_path,
    batch_size=batch_size,
    optimizer=optimizer,
    arch='resnet50',
    frac_train_images=frac_train_images
)`
6. Begin Training & Finetuning 
`trainer = pl.Trainer(max_epochs=hparams.max_epochs,
                     progress_bar_refresh_rate=20,
                     gpus=1,
                     logger=comet_logger,
                     callbacks=[checkpoint_callback, 
                                UnFreezeCallback(), 
                                lr_logger ])
trainer.fit(model_tuned, datamodule=oct_data)`

## Saving & Loading
To load the pre-trained OCTModel use the link: https://drive.google.com/file/d/1MPR5ZQUeLjkwKvEhu2_MuMLL91wirYeR/view?usp=sharing
* Step 1
* Step 2
## Serving the model
* Step 1
* Step 2

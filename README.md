# Self-supervised Approach for Retinal OCT Scan Classification using SimCLR

![alt text](https://mbzuai.ac.ae/application/themes/mbzuai/dist/images/mbzuai_logo.png)

Authors: Diego Saenz, Faris AlMalik, Abdulwahab Sahyoun 



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdu355/ml701_project_grp20/blob/main/ml701_Proj_Final.ipynb)

## Imports & Initialization
1. Install packages with pip: 
  * `!pip install pytorch-lightning image comet-ml captum flask_compress Pillow`
  * `!pip install git+https://github.com/PytorchLightning/pytorch-lightning-bolts.git@master --upgrade`
2. Import all libraries from "Importing libraries and Dataset" section in the notebook.

## Finetuning & Evaluation
1. Download the following datasets & add them to your Google Drive for loading:
  - Mendeley training and testing dataset (Kermany et al. [8]): [Train & Test Dataset link](https://drive.google.com/drive/folders/1YBqEoQSwSlyB_m4f8TeyV9BpkeSK46Ne?usp=sharing)
  - Duke University testing dataset (Srinivasan et al. [14]) + Github testing dataset (Wintergerst et al. [16]): [Test Dataset link](https://drive.google.com/drive/folders/1SOjrG_85785TzWiXE-usTblIW9f4EX_0?usp=sharing)
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

### Evalutation
For evaluating model performance, head to the "Evaluation" section in the notebook:
1. Define `evaluate` function
2. run the sklearn methods: `confusion_matrix` and `classification_report`. Ignore `comet_logger` if it is not used.

## Saving & Loading
To load the pre-trained OCTModel use the link: https://drive.google.com/file/d/1MPR5ZQUeLjkwKvEhu2_MuMLL91wirYeR/view?usp=sharing

### Saving:
1. The model is saved in 2 formats after training is complete using `trainer.save_checkpoint('octmodel.ckpt')` and `torch.save(model_tuned.state_dict(), 'octmodel')`
 * For additional finetuning, use the checkpoint format `octmodel.ckpt` to load the model and continue training. 
 * For serving the model, use the `octmodel` zipfile-based format output by `torch.save`.

### Loading: 
Refer to "Load Trained Model" section in the notebook.
1. Copy the model to your GDrive: https://drive.google.com/file/d/1MPR5ZQUeLjkwKvEhu2_MuMLL91wirYeR/view?usp=sharing
2. Mount GDrive
`from google.colab import drive
drive.mount('/content/drive')`
3. Initialize the model from the pth file`model_loaded = OCTModel(hparams)`


## Serving the model
Make sure `model_loaded` is initialized by Loading the model from the previous section. Refer to "Load Trained Model" section in the notebook.
1. `!pip install flask-ngrok flask==0.12.2 Werkzeug==0.16.1`
2. Run the Flask server at the end of the notebook under "Serve model" section
3. Fork or import this repo https://github.com/abdu355/oct-app
4. Create a streamlit account (https://share.streamlit.io/) and create a new app. Use `oct-app/main/streamlit/ui.py` as the main directory.
5. Add the ngrok url of the Flask server to the Streamlit app Secrets file as `fastapi_url = "http://<your_url>.ngrok.io"`
6. Test the app by navigating to `https://share.streamlit.io/<your_repo_name>/oct-app/main/streamlit/ui.py`

# Dataset Download Instructions

To train this model, you need the **New Plant Diseases Dataset (Augmented)** from Kaggle. 

## Option 1: Manual Download (Recommended)
1. Go to: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. Click the **Download** button.
3. Extract the downloaded `archive.zip`.
4. Inside, you will find a folder named `New Plant Diseases Dataset(Augmented)`.
5. Move this folder into the root of this project (next to `train_model.py`) and rename the directory to `dataset` so that the structure looks like this:
   `c:\Users\ROSHAN MISHRA\.gemini\antigravity\playground\ethereal-cassini\dataset\New Plant Diseases Dataset(Augmented)\train`
   `c:\Users\ROSHAN MISHRA\.gemini\antigravity\playground\ethereal-cassini\dataset\New Plant Diseases Dataset(Augmented)\valid`

## Option 2: Kaggle CLI (if you have kaggle.json configured)
If you have your Kaggle credentials set up (`~/.kaggle/kaggle.json`), you can run the following commands in the terminal:

```bash
pip install kaggle
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d dataset/
```

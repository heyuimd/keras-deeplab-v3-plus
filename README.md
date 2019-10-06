# Keras implementation of Deeplabv3+

### 1. Get the code and dataset.
```bash
git clone 'https://github.com/heyuimd/keras-deeplab-v3-plus'
cd keras-deeplab-v3-plus/
git submodule update --init --recursive
```

### 2. Install python modules.
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### 3. Prepare data for training.
```bash
cd CrackForest-dataset
python bin/resize_image.py ../resized
cd ..
```

### 4. Train model.
```bash
python train.py ./resized
```

### 4. Run prediction on the image.
```bash
python predict.py model.h5 crack.jpg result.jpg
```

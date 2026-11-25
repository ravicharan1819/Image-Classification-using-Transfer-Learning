# Image Classification using Transfer Learning (MobileNetV2)

A CNN image classifier built with **TensorFlow / Keras** using **MobileNetV2** pretrained on ImageNet as a feature extractor, with a custom classification head. Trained by default on the [TF Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) dataset (5 classes, ~3,670 photos), and works on any folder-per-class image dataset.

## Approach
1. **Data pipeline** – `image_dataset_from_directory`, 224×224 images, 80/20 train/validation split, `prefetch` and `cache` for speed.
2. **Preprocessing & augmentation** – random flip, rotation, zoom and contrast (training only), then `mobilenet_v2.preprocess_input` to scale pixels to [-1, 1]. Both are layers inside the model, so prediction takes raw images.
3. **Transfer learning**
   - *Phase 1:* MobileNetV2 base frozen; only the head is trained: GlobalAveragePooling → Dropout → Dense(128, ReLU) → Dropout → Dense(softmax). Adam, lr = 1e-3.
   - *Phase 2 (fine-tuning):* the top 30 layers of the base are unfrozen (BatchNorm layers stay frozen) and trained with a low learning rate (1e-5).
4. **Regularisation** – data augmentation, dropout and early stopping (restoring the best weights) to reduce overfitting. Reusing pretrained features also means far fewer parameters have to be learned from a small dataset.
5. **Evaluation** – accuracy and loss curves for both phases, a per-class classification report and a confusion matrix.

## Project structure
```
image-classification-transfer-learning/
├── train.py          # data loading, two-phase training, evaluation, plots
├── predict.py        # classify new images
├── src/model.py      # MobileNetV2 + custom head, augmentation, fine-tuning helper
├── tests/            # model construction test
├── models/           # saved .keras model + class names (created by train.py)
└── reports/          # training_curves.png, confusion_matrix.png, metrics.json
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py                               # downloads TF Flowers automatically
python train.py --data_dir data/my_images     # or your own dataset (one sub-folder per class)
python predict.py some_flower.jpg
```
A GPU helps but isn't required; on CPU, training the default setup takes roughly 15–30 minutes. Google Colab works well too.

## Results
Fill in from `reports/metrics.json` after training:

| Phase | Validation accuracy | Validation loss |
|---|---|---|
| Frozen base (head only) | | |
| After fine-tuning | | |

![Training curves](reports/training_curves.png)

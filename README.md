# Deep CNN Features → Multinomial Logistic Regression → ROC (from scratch)

![PyTorch](https://img.shields.io/badge/PyTorch-torchvision-ee4c2c)
![Type](https://img.shields.io/badge/graduate%20coursework-UC%20Riverside-lightgrey)

This repo is a three-part pipeline on MNIST. It extracts deep features with a pretrained ResNet-50, trains a multinomial logistic-regression classifier on those features with hand-written gradient descent, and evaluates it with an ROC curve computed without library helpers.

## Parts

1. **Feature extraction.** MNIST digits are passed through an ImageNet-pretrained ResNet-50 (torchvision) with the classification head removed. The penultimate-layer features and labels are saved to `mnist_train.mat` / `mnist_test.mat`.
2. **Multinomial logistic regression.** A softmax classifier is trained on the extracted features with mini-batch gradient descent implemented directly with tensors. Batch size and learning rate are configurable, and train/test accuracy is tracked per epoch.
3. **ROC from scratch.** The true- and false-positive rates are swept over score thresholds by hand to produce the ROC curve for one digit class (class 9).

## Files

| File | Content |
|---|---|
| `CNN.ipynb` | All three parts, with plots |
| `Report.pdf` | Write-up with results and discussion |

## Run

```bash
pip install torch torchvision scipy numpy matplotlib
jupyter notebook CNN.ipynb      # MNIST is downloaded automatically into ./data
```

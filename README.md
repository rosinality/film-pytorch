# film-pytorch
Just another implementation of FiLM (https://arxiv.org/abs/1709.07871) in PyTorch

Requirements:
* Python 3.6
* PyTorch
* torch-vision
* Pillow
* nltk
* tqdm

To train:

1. Download and extract CLEVR v1.0 dataset from http://cs.stanford.edu/people/jcjohns/clevr/
2. Preprocessing question data
```
python preprocess.py [CLEVR directory]
```
3. Run train.py
```
python train.py [CLEVR directory]
```

Learning is still in progress, but already got promising results. (about ~95% accuracy on test set at epoch 21) Surprisingly simple but effective model!

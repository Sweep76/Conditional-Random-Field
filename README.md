# Conditional-Random-Field
model for sequence labeling tasks, such as part-of-speech tagging, using features derived from input sequences and their contexts. It includes methods for training the model using stochastic gradient descent, making predictions, and evaluating performance on training and development datasets.

# Conditional Random Field

## Structure

```sh
.
├── bigdata
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── data
│   ├── dev.conll
│   └── train.conll
├── results
│   ├── acrf.txt
│   ├── bacrf.txt
│   ├── bcrf.txt
│   ├── boacrf.txt
│   ├── bocrf.txt
│   ├── crf.txt
│   ├── oacrf.txt
│   └── ocrf.txt
├── config.py
├── crf.py
├── ocrf.py
├── README.md
└── run.py
```

## Usage

```sh
$ python run.py -h
usage: run.py [-h] [--bigdata] [--anneal] [--optimize] [--regularize]
              [--seed SEED] [--file FILE]

Create Conditional Random Field(CRF) for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --bigdata, -b         use big data
  --anneal, -a          use simulated annealing
  --optimize, -o        use feature extracion optimization
  --regularize, -r      use L2 regularization
  --seed SEED, -s SEED  set the seed for generating random numbers
  --file FILE, -f FILE  set where to store the model
# e.g. Feature extraction optimization + simulated annealing
$ python run.py -b --optimize --anneal
```

## Results

| Large Dataset | Feature Extraction Optimization | Simulated Annealing | Iterations |  dev/P   |  test/P  |     mT(s)      |
| :-----------: | :---------------------------: | :----------------: | :--------: | :------: | :------: | :------------: |
|      ×       |              ×                |         ×         |   12/18    | 88.6405% |    *     | 0:00:52.687575 |
|      ×       |              ×                |         √         |   10/16    | 88.6504% |    *     | 0:00:52.967660 |
|      ×       |              √                |         ×         |   10/16    | 88.9425% |    *     | 0:00:16.543064 |
|      ×       |              √                |         √         |   12/18    | 88.9247% |    *     | 0:00:17.004330 |
|      √       |              ×                |         ×         |   14/25    | 93.8587% | 93.7054% | 0:57:13.132850 |
|      √       |              ×                |         √         |   37/48    | 94.2190% | 93.9665% | 0:56:30.033807 |
|      √       |              √                |         ×         |   15/26    | 93.9705% | 93.8537% | 0:13:33.869449 |
|      √       |              √                |         √         |   13/24    | 94.2107% | 94.0425% | 0:13:25.669687 |

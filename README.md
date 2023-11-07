# Common NN training workflow

Based on this [project](https://github.com/jadore801120/attention-is-all-you-need-pytorch). Further information is needed.

## Model Zoo

1. Dynamic-Weight-Generation-based FullyNN variant  (Deprecated)
2. FullyNN (Omi et al. Done)
3. CTLSTM (Mei et al. Deprecated)
4. RMTPP (Du et al. Done)
5. ifl-tpp (Shchur et al. Done)
6. NeuralODE (Chen et al. Done)
7. NeuralEventODE (Chen et al. Planning)
8. NeuralJSDE (Jia et al. Planning)
9. Transformer TPP (Zuo et al. Done)
10. NCE-tpp (Mei et al. Planning)
11. STRODE (Huang et al. Planning)
12. NSMTPP (Zhu et al. Planning)

## Additional

1. RMTPP can not model the hawkes_2 process and there is no way to resolve it.

## Required functionalities

1. Background training with logs
2. Refurbish current implementations and framework.
3. More intelligent and extensible synthetic data generator.
4. Exclude data files from the git repository.
5. Automantic procedure selection.(WIP)
6. Turn num_event into a dataset information card.
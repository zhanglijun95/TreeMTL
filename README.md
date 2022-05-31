# A Tree-Structured Multi-Task Model Recommender

This is the website for our paper "A Tree-Structured Multi-Task Model Recommender", which is accepted by [AutoML-Conf 2022](https://automl.cc/).
The arXiv version can be found [here](https://arxiv.org/pdf/2203.05092.pdf).

### Abstract

Tree-structured multi-task architectures have been employed to jointly tackle multiple vision tasks in the context of multi-task learning (MTL). The major challenge is to determine where to branch out for each task given a backbone model to optimize for both task accuracy and computation efficiency. To address the challenge, this paper proposes a recommender that, given a set of tasks and a convolutional neural network-based backbone model, automatically suggests tree-structured multi-task architectures that could achieve a high task performance while meeting a user-specified computation budget without performing model training. Extensive evaluations on popular MTL benchmarks show that the recommended architectures could achieve competitive task accuracy and computation efficiency compared with state-of-the-art MTL methods.

![overview](https://github.com/zhanglijun95/TreeMTL/blob/master/assets/overview.jpg)

### Cite

Welcome to cite our work if you find it is helpful to your research.

```
@article{zhang2022tree,
  title={A Tree-Structured Multi-Task Model Recommender},
  author={Zhang, Lijun and Liu, Xiao and Guan, Hui},
  journal={arXiv preprint arXiv:2203.05092},
  year={2022}
}
```

# Description

### Environment

You can build on your conda environment from the provided ```environment.yml```. Feel free to change the env name in the file.

```bash
conda env create -f environment.yml
```

### File Structure

```bash
├── data
│   ├── *_dataloader.py
│   ├── pixel2pixel_loss/metrics.py
├── main
│   ├── layout.py
│   ├── algorithms.py
│   ├── auto_models.py
│   ├── trainer.py
├── models
│   ├── *.prototxt
├── 2task
└── └── *.xlsx
```

### Code Description

The core code are in the folder ```main/```. Specifically,

* ```layout.py```: class of the multi-task task model abstraction **layout**
* ```algorithms.py```: all the important algorithm functions, including the **design space enumerator** and the **task accuracy estimator**.
* ```auto_model.py```: class of the backbone model sequentialized from the given ```*.prototxt```, namely the **branching point detector**.

Other folders are:

* ```data/```: dataloader, task loss and metrics for NYUv2 and Taskonomy
* ```main/trainer.py```: trainer functions for model training
* ```models/```: ```*.prototxt``` files for backbone models, Deeplab-ResNet34 and MobileNetV2
* ```2task/```: task accuracy for 2-task models

# How to Use

**Note: Please refer to ```Example.ipynb``` for more details.**

### Branching Point Detector

Given a backbone model specified in the format of ```*.prototxt```, the branching point detector will automatically divide it into sequential blocks.

```bash
prototxt = 'models/deeplab_resnet34_adashare.prototxt' 
backbone = MTSeqBackbone(prototxt)
B = len(backbone.basic_blocks)
```

The user can __further specify coarse-grained branching points__ based on the auto-generated branching points by defining a mapping dictionary.

```bash
coarse_B = 5
mapping = {0:[0], 1:[1,2,3], 2:[4,5,6,7], 3:[8,9,10,11,12,13], 4:[14,15,16], 5:[17]}
```

### Design Spce Enumerator

Given the number of tasks $T$ and the number of branching points $B$, the design space enumerator could explore the tree-structured multi-task model architectures space completely.

```bash
T = 3 
layout_list = enumerator(T, coarse_B)
```

### Task Accuracy Estimator

For each layout in the design space, we will estimate its task accuracy from the task accuacy of associated 2-task models.

##### Step 1: Load 2-task models results

__The task accuracy of all the 2-task models should be stored in excel and organized as examples in ```2task/*.xlsx```.__

```bash
two_task_pd = pd.read_excel('2task/NYUv2_2task_metrics_resnet_1129_val_acc.xlsx',engine='openpyxl',index_col=0)
```

##### Step 2: Compute score weights

The 2-task results will be reorganized by ```reorg_two_task_results``` for the further computation, and the task weights for the layout scores are computed by ```compute_weights```.

```bash
two_task_metrics = reorg_two_task_results(two_task_pd, T, coarse_B)
score_weights = compute_weights(two_task_pd, T)
```

##### Step 3: Compute layout score from accociated 2-task models

For each layout in the design space ```layout_list```, figure out the accociated 2-task models by ```metric_inference```, then set the final score by ```L.set_score_weighted```.

```bash
# Run for all L
for L in layout_list:
    subtree = metric_inference(L, two_task_metrics)
    L.set_score_weighted(score_weights)
```

##### Step 4: Sort the layouts

```bash
layout_order = sorted(range(len(layout_list)), key=lambda k: layout_list[k].score,reverse=True)
```

### Appendix

* We provide **dataloader, loss, and metrics** for the two datasets, NYUv2 [[1]](#1) and Taskonomy  [[2]](#2). You can download NYUv2 [here](https://drive.google.com/drive/folders/1KX9chooxefvrZACtFR441ShdkbS3F2lt?usp=sharing). For Tiny-Taskonomy, you will need to contact the authors directly. See their [official website](http://taskonomy.stanford.edu/).
* We further provide **a 2-task and n-task model generator**, that can automatically build up the 2-task models based on the given branching points, and the n-task models based on the given layout. The automation backend is AutoMTL [[3]](#3).
* We also provide **a trainer tool** to cover the 2-task and n-task models' training procedure.

Please refer to ```Example.ipynb``` for the usage.

# References

<a id="1">[1]</a>
Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob.
Indoor segmentation and support inference from rgbd images.
ECCV, 746-760, 2012.

<a id="2">[2]</a>
Zamir, Amir R and Sax, Alexander and Shen, William and Guibas, Leonidas J and Malik, Jitendra and Savarese, Silvio.
Taskonomy: Disentangling task transfer learning.
CVPR, 3712-3722, 2018.

<a id="3">[3]</a>
Zhang L, Liu X, Guan H. AutoMTL: A Programming Framework for Automated Multi-Task Learning. arXiv preprint arXiv:2110.13076, 2021.

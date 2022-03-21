# A Tree-Structured Multi-Task Model Recommender 
This is the website for our paper "A Tree-Structured Multi-Task Model Recommender". 
The arXiv version can be found [here](https://arxiv.org/pdf/2203.05092.pdf).

### Abstract
Tree-structured multi-task architectures have been employed to jointly tackle multiple vision tasks in the context of multi-task learning (MTL). The major challenge is to determine where to branch out for each task given a backbone model to optimize for both task accuracy and computation efficiency. To address the challenge, this paper proposes a recommender that, given a set of tasks and a convolutional neural network-based backbone model, automatically suggests tree-structured multi-task architectures that could achieve a high task performance while meeting a user-specified computation budget without performing model training. Extensive evaluations on popular MTL benchmarks show that the recommended architectures could achieve competitive task accuracy and computation efficiency compared with state-of-the-art MTL methods.

![overview](https://github.com/zhanglijun95/AutoMTL/blob/main/assets/overview.jpg)

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

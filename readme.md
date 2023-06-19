![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/logo.png)

------

[![ACM SIGSpatial](https://img.shields.io/badge/ACM%20SIGSPATIAL'21-LibCity-orange)](https://dl.acm.org/doi/10.1145/3474717.3483923) [![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/) [![Pytorch](https://img.shields.io/badge/Pytorch-1.7.1%2B-blue)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE.txt) [![star](https://img.shields.io/github/stars/LibCity/Bigscity-LibCity?style=social)](https://github.com/LibCity/Bigscity-LibCity/stargazers) [![fork](https://img.shields.io/github/forks/LibCity/Bigscity-LibCity?style=social)](https://github.com/LibCity/Bigscity-LibCity/network/members) 

# LibCity（阡陌）

[HomePage](https://libcity.ai/)|[Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/index.html)|[Datasets](https://github.com/LibCity/Bigscity-LibCity-Datasets)|[Paper Conference](https://dl.acm.org/doi/10.1145/3474717.3483923)|[Paper Journal](https://arxiv.org/abs/2304.14343)|[Paper List](https://github.com/LibCity/Bigscity-LibCity-Paper)|[Experiment Tool](https://github.com/LibCity/Bigscity-LibCity-WebTool)|[中文版](https://github.com/LibCity/Bigscity-LibCity/blob/master/readme_zh.md) 

LibCity is a unified, comprehensive, and extensible library, which provides researchers with a credible experimental tool and a convenient development framework in the traffic prediction field. Our library is implemented based on PyTorch and includes all the necessary steps or components related to traffic prediction into a systematic pipeline, allowing researchers to conduct comprehensive experiments. Our library will contribute to the standardization and reproducibility in the field of traffic prediction.

LibCity currently supports the following tasks:

* Traffic State Prediction
  * Traffic Flow Prediction
  * Traffic Speed Prediction
  * On-Demand Service Prediction
  * Origin-destination Matrix Prediction
  * Traffic Accidents Prediction
* Trajectory Next-Location Prediction
* Estimated Time of Arrival
* Map Matching
* Road Network Representation Learning

## Features

* **Unified**: LibCity builds a systematic pipeline to implement, use and evaluate traffic prediction models in a unified platform. We design basic spatial-temporal data storage, unified model instantiation interfaces, and standardized evaluation procedure.

* **Comprehensive**: 65 models covering 9 traffic prediction tasks have been reproduced to form a comprehensive model warehouse. Meanwhile, LibCity collects 35 commonly used datasets of different sources and implements a series of commonly used evaluation metrics and strategies for performance evaluation. 

* **Extensible**: LibCity enables a modular design of different components, allowing users to flexibly insert customized components into the library. Therefore, new researchers can easily develop new models with the support of LibCity.

## LibCity News

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif)**06/20/2023:** We released the Beijing trajectory dataset collected in November 2015, including 1018312 trajectories. We obtained the corresponding road network data from OpenStreetMap and preprocessed the trajectory data to get the Beijing trajectory dataset matched to the road network, and we believed that this dataset could promote the development of urban trajectory mining tasks. Please refer to this [link](https://github.com/aptx1231/START) to obtain it and ensure that this data is **used for research purposes only**. 

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **06/04/2023**: LibCity won the second prize in the 3rd China Science Open Source Software Creativity Competition! [Weixin](https://mp.weixin.qq.com/s?__biz=MzA3NzM4OTc4Mw==&mid=2454775999&idx=1&sn=881a31468c5cd472ed72967b487837cf&chksm=88f68207bf810b1157ac622ae0beba0a1f2ca8ece38fa5c743c4e082c30d9e27d23b92b61530&scene=126&sessionid=1687198811#rd)

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **04/27/2023**: We published a [long paper](https://arxiv.org/abs/2304.14343) on LibCity, including (1) classification and base units of urban spatial-temporal data and proposed a unified storage format, i.e., atomic files, (2) a detailed review of urban spatial-temporal prediction field (including macro-group prediction, micro-individual prediction, and fundamental tasks), (3) proposed LibCity, an open source library for urban spatial-temporal prediction, detailing each module and use cases, and providing a web-based experiment management and visualization platform, (4) selected more than 20 models and datasets for comparison experiments based on LibCity, obtained model performance rankings and summarized promising future research directions. Please check this [link](https://arxiv.org/abs/2304.14343) for more details.

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **11/19/2022**: Our self-attention-based traffic flow prediction model **PDFormer** developed on LibCity was accepted by **AAAI2023**, please check this [link](https://github.com/BUAABIGSCity/PDFormer) for more details.

[![new](https://github.com/RUCAIBox/RecBole/raw/master/asset/new.gif)](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **08/05/2022**: We develop an **experiment management tool** for the LibCity, which allows users to complete the experiments in a visual interface. The link to the code repository is [here](https://github.com/LibCity/Bigscity-LibCity-WebTool). Some introduction (in Chinese): [Weixin](https://mp.weixin.qq.com/s?__biz=MzA3NzM4OTc4Mw==&mid=2454773897&idx=1&sn=e09cc3fc7dd772a579dd10730f8fadd8&chksm=88f68a31bf810327849442c6af4bf59d5042dfb9871247239a49f070dbeb9f321b41706da157&scene=126&&sessionid=1669002707#rd), [Zhihu](https://zhuanlan.zhihu.com/p/550605104)

**04/27/2022**: We release the first version of LibCity **v0.3**, the latest version, supporting 9 types of spatio-temporal prediction tasks, covering more than 60 prediction models and nearly 40 urban spatio-temporal datasets.

**11/24/2021**: We provide some introductory tutorials (in Chinese) on Zhihu, [link1](https://zhuanlan.zhihu.com/p/401186930), [link2](https://zhuanlan.zhihu.com/p/400814990), [link3](https://zhuanlan.zhihu.com/p/400819354), [link4](https://zhuanlan.zhihu.com/p/400821482), [link5](https://zhuanlan.zhihu.com/p/401190615), [link6](https://zhuanlan.zhihu.com/p/436191860)....

**11/10/2021**: We provide a document that describes in detail the format of the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) defined by LibCity. You can download [English Version](https://libcity.ai/A-Unified-Storage-Format-of-Traffic-Data-Atomic-Files-in-LibCity.pdf) and [Chinese Version](https://libcity.ai/LibCity%E4%B8%AD%E4%BA%A4%E9%80%9A%E6%95%B0%E6%8D%AE%E7%BB%9F%E4%B8%80%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F-%E5%8E%9F%E5%AD%90%E6%96%87%E4%BB%B6.pdf) here for details.

**11/07/2021**: We have a presentation on ACM SIGSPATIAL 2021 Local Track to introduce LibCity. You can download [LibCity Presentation Slide(Chinese)](https://libcity.ai/LibCity-城市时空预测深度学习开源平台.pdf) and [LibCity Chinese Tutorial](https://libcity.ai/LibCity-中文Tutorial.pptx) here.

**11/07/2021**: We have a presentation on ACM SIGSPATIAL 2021 Main Track to introduce LibCity. Here are the [Presentation Video(English)](https://www.bilibili.com/video/BV19q4y1g7Rh/) and the [Presentation Slide(English)](https://libcity.ai/LibCity-Presentation.pdf).

## Overall Framework

![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/framework.png)

* **Configuration Module**: Responsible for managing all the parameters involved in the framework.
* **Data Module**: Responsible for loading datasets and data preprocessing operations.
* **Model Module**: Responsible for initializing the reproduced baseline model or custom model.
* **Evaluation Module**: Responsible for evaluating model prediction results through multiple indicators.
* **Execution Module**: Responsible for model training and prediction.

## Installation

LibCity can only be installed from source code.

Please execute the following command to get the source code.

```shell
git clone https://github.com/LibCity/Bigscity-LibCity
cd Bigscity-LibCity
```

More details about environment configuration is represented in [Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/get_started/install.html).

## Quick-Start

Before run models in LibCity, please make sure you download at least one dataset and put it in directory `./raw_data/`. The dataset link is [BaiduDisk with code 1231](https://pan.baidu.com/s/1qEfcXBO-QwZfiT0G3IYMpQ) or [Google Drive](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing). All dataset used in LibCity needs to be processed into the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) format.

The script `run_model.py` is used for training and evaluating a single model in LibCity. When run the `run_model.py`, you must specify the following three parameters, namely **task**, **dataset** and **model**.  

For example:

```sh
python run_model.py --task traffic_state_pred --model GRU --dataset METR_LA
```

This script will run the GRU model on the METR_LA dataset for traffic state prediction task under the default configuration.  **We have released the correspondence between datasets, models, and tasks at [here](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/dataset_for_task.html).** More details is represented in [Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/get_started/quick_start.html).

## TensorBoard Visualization

During the model training process, LibCity will record the loss of each epoch, and support tensorboard visualization.

After running the model once, you can use the following command to visualize:

```shell
tensorboard --logdir 'libcity/cache'
```

```
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Visit this address([http://localhost:6006/](http://localhost:6006/)) in the browser to see the visualized result.

![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/tensorboard.png)

## Reproduced Model List

For a list of all models reproduced in LibCity, see [Docs](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/model.html), where you can see the abbreviation of the model and the corresponding papers and citations.

## Tutorial

In order to facilitate users to use LibCity, we provide users with some tutorials:

- We gave lectures on both ACM SIGSPATIAL 2021 Main Track and Local Track. For related lecture videos and Slides, please see our [HomePage](https://libcity.ai/#/tutorial) (in Chinese and English).
- We provide entry-level tutorials (in Chinese and English) in the documentation.
  - [Install and quick start](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/install_quick_start.html)  & [安装和快速上手](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/install_quick_start.html)
  - [Run an existing model in LibCity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/run_model.html) & [运行LibCity中已复现的模型](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/run_model.html)
  - [Add a new model to LibCity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/add_model.html)  & [在LibCity中添加新模型](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/add_model.html)
  - [Tuning the model with automatic tool](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/hyper_tune.html) & [使用自动化工具调参](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/hyper_tune.html)
  - [Visualize Atomic Files](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/data_visualization.html) & [原子文件可视化](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/data_visualization.html)
- In order to facilitate the use of domestic users in China, we provide an introductory tutorial (in Chinese) on Zhihu.
  - [LibCity：一个统一、全面、可扩展的交通预测算法库](https://zhuanlan.zhihu.com/p/401186930)
  - [LibCity入门教程（1）——安装与快速上手](https://zhuanlan.zhihu.com/p/400814990)
  - [LibCity入门教程（2）——运行LibCity中已复现的模型](https://zhuanlan.zhihu.com/p/400819354)
  - [LibCity入门教程（3）——在LibCity中添加新模型](https://zhuanlan.zhihu.com/p/400821482)
  - [LibCity入门教程（4）—— 自动化调参工具](https://zhuanlan.zhihu.com/p/401190615)
  - [北航BIGSCity课题组提出LibCity工具库：城市时空预测深度学习开源平台](https://zhuanlan.zhihu.com/p/436191860)

## Contribution

The LibCity is mainly developed and maintained by Beihang Interest Group on SmartCity ([BIGSCITY](https://www.bigcity.ai/)). The core developers of this library are [@aptx1231](https://github.com/aptx1231) and [@WenMellors](https://github.com/WenMellors). 

Several co-developers have also participated in the reproduction of  the model, the list of contributions of which is presented in the [reproduction contribution list](./contribution_list.md).

If you encounter a bug or have any suggestion, please contact us by [raising an issue](https://github.com/LibCity/Bigscity-LibCity/issues). You can also contact us by sending an email to bigscity@126.com.

## Cite

Our paper is accepted by ACM SIGSPATIAL 2021. If you find LibCity useful for your research or development, please cite our [paper](https://dl.acm.org/doi/10.1145/3474717.3483923).

```
@inproceedings{libcity,
  author = {Wang, Jingyuan and Jiang, Jiawei and Jiang, Wenjun and Li, Chao and Zhao, Wayne Xin},
  title = {LibCity: An Open Library for Traffic Prediction},
  year = {2021},
  isbn = {9781450386647},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3474717.3483923},
  doi = {10.1145/3474717.3483923},
  booktitle = {Proceedings of the 29th International Conference on Advances in Geographic Information Systems},
  pages = {145–148},
  numpages = {4},
  keywords = {Spatial-temporal System, Reproducibility, Traffic Prediction},
  location = {Beijing, China},
  series = {SIGSPATIAL '21}
}
```

For the long paper, please cite it as follows:

```
@article{libcitylong,
  title={Towards Efficient and Comprehensive Urban Spatial-Temporal Prediction: A Unified Library and Performance Benchmark}, 
  author={Jingyuan Wang and Jiawei Jiang and Wenjun Jiang and Chengkai Han and Wayne Xin Zhao},
  journal={arXiv preprint arXiv:2304.14343},
  year={2023}
}
```

## License

LibCity uses [Apache License 2.0](https://github.com/LibCity/Bigscity-LibCity/blob/master/LICENSE.txt).

## Stargazers

[![Stargazers repo roster for @LibCity/Bigscity-LibCity](https://reporoster.com/stars/LibCity/Bigscity-LibCity)](https://github.com/LibCity/Bigscity-LibCity/stargazers)

## Forkers

[![Forkers repo roster for @LibCity/Bigscity-LibCity](https://reporoster.com/forks/LibCity/Bigscity-LibCity)](https://github.com/LibCity/Bigscity-LibCity/network/members)
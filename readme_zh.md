![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/logo.png)

------

# LibCity（阡陌）

[主页](https://libcity.ai/)|[文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/#)|[数据集](https://github.com/LibCity/Bigscity-LibCity-Datasets)|[论文列表](https://github.com/LibCity/Bigscity-LibCity-Paper)|[英文版](https://github.com/LibCity/Bigscity-LibCity/blob/master/readme.md)

LibCity 是一个统一、全面、可扩展的代码库，为交通预测领域的研究人员提供了一个可靠的实验工具和便捷的开发框架。 我们的库基于 PyTorch 实现，并将与交通预测相关的所有必要步骤或组件包含到系统的流水线中，使研究人员能够进行全面的对比实验。 我们的库将有助于交通预测领域的标准化和可复现性。

LibCity 目前支持以下任务：

* 交通状态预测
  * 交通流量预测
  * 交通速度预测
  * 交通需求预测
  * 起点-终点（OD）矩阵预测
  * 交通事故预测
* 轨迹下一跳预测
* 到达时间预测
* 路网匹配
* 路网表征学习

## Features

* **统一性**：LibCity 构建了一个系统的流水线以在一个统一的平台上实现、使用和评估交通预测模型。 我们设计了统一的时空数据存储格式、统一的模型实例化接口和标准的模型评估程序。

* **全面性**：复现覆盖 9 个交通预测任务的 60 个模型，形成了全面的模型库。 同时，LibCity 收集了 35 个不同来源的常用数据集，并实现了一系列常用的性能评估指标和策略。

* **可扩展性**：LibCity 实现了不同组件的模块化设计，允许用户灵活地加入自定义组件。 因此，新的研究人员可以在 LibCity 的支持下轻松开发新模型。

## Overall Framework

![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/framework.png)

* **Configuration Module**: 负责管理框架中涉及的所有参数。
* **Data Module**: 负责加载数据集和数据预处理操作。
* **Model Module**: 负责初始化基线模型或自定义模型。
* **Evaluation Module**: 负责通过多个指标评估模型预测结果。
* **Execution Module**: 负责模型训练和预测。

## Installation

LibCity 只能从源代码安装。

请执行以下命令获取源代码。

```shell
git clone https://github.com/LibCity/Bigscity-LibCity
cd Bigscity-LibCity
```

有关环境配置的更多详细信息，请参见 [文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/get_started/install.html).

## Quick-Start

在 LibCity 中运行模型之前，请确保您至少下载了一个数据集并将其放在目录 `./raw_data/` 中。 数据集链接是 [BaiduDisk with code 1231](https://pan.baidu.com/s/1qEfcXBO-QwZfiT0G3IYMpQ) 或 [Google Drive](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing) 。LibCity 中所用的数据集需要被处理成[原子文件](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/user_guide/data/atomic_files.html)的格式。

脚本 `run_model.py` 用于在 LibCity 中训练和评估单个模型。 运行`run_model.py`时，必须指定以下三个参数，即**task、dataset和model**。例如：

```sh
python run_model.py --task traffic_state_pred --model GRU --dataset METR_LA
```

该脚本将在默认配置下，在 METR_LA 数据集上运行 GRU 模型，以进行交通状态预测任务。目前我们已经在 [文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/user_guide/data/dataset_for_task.html) 发布了数据集、模型和任务之间的对应关系表格供用户参考。

更多细节请访问 [文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/get_started/quick_start.html) 。

## TensorBoard Visualization

在模型训练过程中，LibCity 会记录每个 epoch 的损失，并支持 tensorboard 可视化。

模型运行一次后，可以使用以下命令进行可视化：

```shell
tensorboard --logdir 'libcity/cache'
```

```
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

在浏览器中访问这个地址（[http://localhost:6006/](http://localhost:6006/)） 可以看到可视化的结果。

![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/tensorboard.png)

## Reproduced Model List

LibCity 中所复现的全部模型列表见[文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/user_guide/model.html)，在这里你可以看到模型的简称和对应的论文及引用文献。

## Tutorial

为了方便用户使用 LibCity，我们为用户提供了一些入门教程：

- 我们在 ACM SIGSPATIAL 2021 Main Track 以及 Local Track 上都进行了演讲，相关的演讲视频和Slide见我们的[主页](https://libcity.ai/#/tutorial)（中英文）。
- 我们在文档中提供了入门级教程（中英文）。
  - [Install and quick start](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/install_quick_start.html)  & [安装和快速上手](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/install_quick_start.html)
  - [Run an existing model in LibCity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/run_model.html) & [运行LibCity中已复现的模型](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/run_model.html)
  - [Add a new model to LibCity](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/add_model.html)  & [在LibCity中添加新模型](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/add_model.html)
  - [Tuning the model with automatic tool](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/hyper_tune.html) & [使用自动化工具调参](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/hyper_tune.html)
  - [Visualize Atomic Files](https://bigscity-libcity-docs.readthedocs.io/en/latest/tutorial/data_visualization.html) & [原子文件可视化](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/tutorial/data_visualization.html)
- 为了便于国内用户使用，我们在知乎上提供了入门教程（中文）。
  - [LibCity：一个统一、全面、可扩展的交通预测算法库](https://zhuanlan.zhihu.com/p/401186930)
  - [LibCity入门教程（1）——安装与快速上手](https://zhuanlan.zhihu.com/p/400814990)
  - [LibCity入门教程（2）——运行LibCity中已复现的模型](https://zhuanlan.zhihu.com/p/400819354)
  - [LibCity入门教程（3）——在LibCity中添加新模型](https://zhuanlan.zhihu.com/p/400821482)
  - [LibCity入门教程（4）—— 自动化调参工具](https://zhuanlan.zhihu.com/p/401190615)
  - [北航BIGSCity课题组提出LibCity工具库：城市时空预测深度学习开源平台](https://zhuanlan.zhihu.com/p/436191860)

## Contribution

LibCity 主要由北航智慧城市兴趣小组 ([BIGSCITY](https://www.bigcity.ai/)) 开发和维护。 该库的核心开发人员是 [@aptx1231](https://github.com/aptx1231) 和 [@WenMellors](https://github.com/WenMellors)。

若干共同开发者也参与了模型的复现，其贡献列表在 [贡献者列表](./contribution_list.md) 。

如果您遇到错误或有任何建议，请通过 [提交issue](https://github.com/LibCity/Bigscity-LibCity/issues) 的方式与我们联系。您也可以通过发送邮件的方式联系我们，邮箱为bigscity@126.com。

## Cite

该工作已被ACM SIGSPATIAL 2021接收。如果您认为LibCity对您的科研工作有帮助，请引用我们的[论文](https://dl.acm.org/doi/10.1145/3474717.3483923)。

```
@inproceedings{10.1145/3474717.3483923,
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

```
Jingyuan Wang, Jiawei Jiang, Wenjun Jiang, Chao Li, and Wayne Xin Zhao. 2021. LibCity: An Open Library for Traffic Prediction. In Proceedings of the 29th International Conference on Advances in Geographic Information Systems (SIGSPATIAL '21). Association for Computing Machinery, New York, NY, USA, 145–148. DOI:https://doi.org/10.1145/3474717.3483923
```

## License

LibCity 遵循 [Apache License 2.0](https://github.com/LibCity/Bigscity-LibCity/blob/master/LICENSE.txt) 协议。

## Stargazers

[![Stargazers repo roster for @LibCity/Bigscity-LibCity](https://reporoster.com/stars/LibCity/Bigscity-LibCity)](https://github.com/LibCity/Bigscity-LibCity/stargazers)

## Forkers

[![Forkers repo roster for @LibCity/Bigscity-LibCity](https://reporoster.com/forks/LibCity/Bigscity-LibCity)](https://github.com/LibCity/Bigscity-LibCity/network/members)
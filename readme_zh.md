![](https://bigscity-libcity-docs.readthedocs.io/en/latest/_images/logo.png)

------

# LibCity（阡陌）

[主页](https://libcity.ai/)|[文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/#)|[数据集](https://github.com/LibCity/Bigscity-LibCity-Datasets)|[论文列表](https://github.com/LibCity/Bigscity-LibCity-Paper)|[英文版](https://github.com/LibCity/Bigscity-LibCity/blob/master/readme.md)

LibCity 是一个统一、全面、可扩展的代码库，为交通预测领域的研究人员提供了一个可靠的实验工具和便捷的开发框架。 我们的库基于 PyTorch 实现，并将与交通预测相关的所有必要步骤或组件包含到系统的流水线中，使研究人员能够进行全面的对比实验。 我们的库将有助于交通预测领域的标准化和可复现性。

LibCity 目前支持以下任务：

* 时间序列预测
* 交通状态预测
  * 交通流量预测
  * 交通速度预测
  * 交通需求预测
  * OD矩阵预测
* 轨迹下一跳预测
* 路网匹配
* 路网表征学习

## Features

* **统一性**：LibCity 构建了一个系统的流水线以在一个统一的平台上实现、使用和评估交通预测模型。 我们设计了统一的时空数据存储格式、统一的模型实例化接口和标准的模型评估程序。

* **全面性**：复现覆盖 8 个交通预测任务的 54 个模型，形成了全面的模型库。 同时，LibCity 收集了 32 个不同来源的常用数据集，并实现了一系列常用的性能评估指标和策略。

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

在 LibCity 中运行模型之前，请确保您至少下载了一个数据集并将其放在目录 `./raw_data/` 中。 数据集链接是 [BaiduDisk with code 1231](https://pan.baidu.com/s/1qEfcXBO-QwZfiT0G3IYMpQ) 或 [Google Drive](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing) 。

脚本 `run_model.py` 用于在 LibCity 中训练和评估单个模型。 运行`run_model.py`时，必须指定以下三个参数，即**task、dataset和model**。例如：

```sh
python run_model.py --task traffic_state_pred --model GRU --dataset METR_LA
```

该脚本将在默认配置下，在 METR_LA 数据集上运行 GRU 模型，以进行交通状态预测任务。目前我们已经在 [文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/user_guide/data/dataset_for_task.html) 发布了数据集、模型和任务之间的对应关系表格供用户参考。

更多细节请访问 [文档](https://bigscity-libcity-docs.readthedocs.io/zh_CN/latest/get_started/quick_start.html) 。

## Contribution

LibCity 主要由北航智慧城市兴趣小组 ([BIGSCITY](https://www.bigcity.ai/)) 开发和维护。 该库的核心开发人员是 [@aptx1231](https://github.com/aptx1231) 和 [@WenMellors](https://github.com/WenMellors)。

若干共同开发者也参与了模型的复现，其贡献列表在 [贡献者列表](./contribution_list.md) 。

如果您遇到错误或有任何建议，请通过以下方式与我们联系： [提交issue](https://github.com/LibCity/Bigscity-LibCity/issues)。

## Cite

该工作已被ACM SIGSPATIAL 2021接收。如果您认为LibCity对您的科研工作有帮助，请引用我们的[论文](https://libcity.ai/#/paper)。

```
@proceedings{libcity,
  editor={Jingyuan Wang and Jiawei Jiang and Wenjun Jiang and Chao Li and Wayne Xin Zhao},
  title={LibCity: An Open Library for Traffic Prediction},
  booktitle={{SIGSPATIAL} '21: 29th International Conference on Advances in Geographic Information Systems, Beijing, China, November 2-5, 2021 },
  publisher={{ACM}},
  year={2021}
}
```

```
Jingyuan Wang, Jiawei Jiang, Wenjun Jiang, Chao Li, and Wayne Xin Zhao. 2021. LibCity: An Open Library for Traffic Prediction. In Proceedings of the 29th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems. 
```

## License

LibCity 遵循 [Apache License 2.0](https://github.com/LibCity/Bigscity-LibCity/blob/master/LICENSE.txt) 协议。


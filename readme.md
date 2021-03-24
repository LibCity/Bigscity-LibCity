## 介绍

[文档](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/index.html)

本项目为交通大数据领域下的模型开发开源框架，目前支持以下任务：

* 交通轨迹下一跳预测
* 交通状态预测

#### 架构说明

本框架主要由 Config、Data、Model、Executor、Evaluator、Pipeline 六个模块构成。其中每个模块的职责如下所述：

* Config：负责统一管理参数配置。
* Data：负责加载数据集，并进行数据预处理与数据集划分。
* Model：负责具体模型的实现。
* Executor：负责运行模型，如：训练模型、模型预测。
* Evaluator：负责评估模型。
* Pipeline：将前五个模块整合起来，以流水线的方式先后执行完成用户指定的任务。

#### 流水线实例：运行单个模型流水线

![](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/_images/pipeline.png)

1. 初始化流水线配置。用户可通过命令行传参与指定 config 文件的方式灵活地调整流水线参数配置。（依托 Config 模块）
2. 数据集加载与数据预处理，数据转换并划分训练集、验证集、测试集。（依托 Data 模块）
3. 加载模型。（依托 Model 模块）
4. 训练验证模型，并在测试集上进行测试。（依托 Executor 模块）
5. 评估模型测试输出。（依托 Evaluator 模块）

> ps: 由于模型验证时就需要 Evaluator 来进行评估，因此实际实现中，Evaluator 模块是由 Executor 模块进行实例化并调用的。

## 快速开始

#### 运行单个模型

框架根目录下提供运行单个模型的脚本 run_model.py，并提供一系列命令行参数以允许用户能够调整流水线参数配置。命令行运行示例：

```sh
python run_model.py --task traj_loc_pred --model DeepMove --dataset foursquare_tky
```

所支持的命令行参数如下：：

- `task`：所要执行的任务名，包括`traj_loc_pred`和`traffic_state_pred`，默认为`traj_loc_pred`。
- `model`：所要运行的模型名，应是`trafficdl/model/`目录下各Model类名中的一个，默认为`DeepMove`。
- `dataset`：所要运行的数据集，默认为 `foursquare_tky`。
- `config_file`：用户指定 config 文件名，默认为 `None`。
- `saved_model`：是否保存训练的模型结果，默认为 `True`。
- `train`：当模型已被训练时是否要重新训练，默认为 `True`。
- `batch_size`：单次输入的 Batch 大小。
- `train_rate`：训练集所占比例，如`0.6`，划分顺序是【训练集，验证集，测试集】。
- `eval_rate`：验证集所占比例，如`0.2`，划分顺序是【训练集，验证集，测试集】。
- `learning_rate`：优化器的学习率。
- `max_epoch`：训练的最大轮次。
- `gpu`：是否是用GPU，默认为 `True`。
- `gpu_id`：指定使用的GPU的id，默认为`0`。

## 标准赛道

在交通大数据领域中，长期存在着评测数据集不统一、评测指标不统一、数据集预处理不统一等现象，导致了不同模型的性能可比性较差。因此本项目为了解决上述问题，为每个任务实现了一套标准流水线（赛道）。

标准赛道上，使用项目提供的原始数据集、标准数据模块（Data 模块）、标准评估模块（Evaluator 模块），从而约束不同模型使用相同的数据输入与评估指标，以提高评估结果的可比性。

下面对不同任务的标准数据输入格式与评估输入格式进行说明：

#### 轨迹下一跳预测

标准数据输入格式为类字典的 [Batch](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/user_guide/data/batch.html) 对象实例，该对象所具有的键名如下：

* `history_loc`：历史轨迹位置信息，`shape = (batch_size, history_len)`， `history_len` 为历史轨迹的长度。

* `history_tim`：历史轨迹时间信息，`shape = (batch_size, history_len)`。

* `current_loc`：表示当前轨迹位置信息，`shape = (batch_size, current_len)`， `current_len` 为历史轨迹的长度。

* `current_tim`：表示当前轨迹位置信息，`shape = (batch_size, current_len)`。

* `uid`：每条轨迹所属用户的 id，`shape = (batch_size)`。

* `target`：期望的下一跳位置，`shape = (batch_size) `。

标准评估输入格式为字典对象，该字典具有的键名如下：

* `uid`：每条输出所属的用户 id，`shape = (batch_size)`。
* `loc_true`：期望下一跳位置信息，`shape = (batch_size)`。
* `loc_pred`：模型预测输出，`shape = (batch_size, output_dim)`。 

#### 交通状态预测

根据交通数据的不同空间结构，交通状态数据一般可以用如下几种格式的张量进行表示：

- `（N,T,F）`的三维张量，`T`是时间长度，`F`是特征维度，`N`是传感器的个数。
- `（T,F,I,J）`的四维张量，`T`是时间长度，`F`是特征维度，`I,J`表示网格数据的行列索引。
- `（T,F,S,T）`的四维张量，`T`是时间长度，`F`是特征维度，`S,T`表示`od`数据的起点和终点的编号。
- `（T,F,SI,SJ,TI,TJ）`的六维张量，`T`是时间长度，`F`是特征维度，`SI,SJ,TI,TJ`表示网格结构的`od`数据的起点和终点的行列索引。

标准模型输入格式为类字典的 [Batch](https://aptx1231.github.io/Bigscity-TrafficDL-Docs/user_guide/data/batch.html) 对象实例，该对象所具有的键名如下：

- `X`：模型输入的多维张量，`shape = (batch_size, T_in, space_dim, feature_dim)`，分别表示 batch 中的样本总数，输入时间窗的宽度，空间上的维度，数据特征维数。其中，空间上的维度可以是上文中的`N`或`I,J`或`S,T`或`SI,SJ,TI,TJ`。
- `y`：模型期望输出的多维张量，`shape = (batch_size, T_out, space_dim, feature_dim)`，分别表示 batch 中的样本总数，输出时间窗的宽度，空间上的维度，数据特征维数。其中，空间上的维度可以是上文中的`N`或`I,J`或`S,T`或`SI,SJ,TI,TJ`。
- `X_ext`：可选的外部数据，`shape = (batch_size, T_in, ext_dim)`，分别表示 batch 中的样本总数，输入时间窗的宽度，空间上的维度，外部数据特征维数。部分模型可能直接将`X_ext`融合到`X`中作为模型的输入。
- `y_ext`：可选的外部数据，`shape = (batch_size, T_out, ext_dim)`，分别表示 batch 中的样本总数，输出时间窗的宽度，空间上的维度，外部数据特征维数。

标准评估模块的输入格式为字典对象，该对象所具有的键名如下：

- `y_true`：真实值，格式同输入中的 `y`。
- `y_pred`：预测值，格式同输入中的 `y`。


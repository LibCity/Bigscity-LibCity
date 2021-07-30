![](https://bigscity-libtraffic-docs.readthedocs.io/en/latest/_images/logo.png)

------

# LibTraffic（阡陌）

[HomePage](https://libtraffic.github.io/Bigscity-LibTraffic-Website)|[Docs](https://bigscity-libtraffic-docs.readthedocs.io/en/latest/index.html)|[Datasets](https://github.com/LibTraffic/Bigscity-LibTraffic-Datasets)|[Paper List](https://github.com/LibTraffic/Bigscity-LibTraffic-Paper) |[中文版](https://github.com/LibTraffic/Bigscity-LibTraffic/blob/master/readme_zh.md)

LibTraffic is a unified, flexible and comprehensive traffic prediction library, which  provides researchers with a credibly experimental tool and a convenient development framework. Our library is implemented based on PyTorch, and includes all the necessary steps or components related to traffic prediction into a systematic pipeline.

LibTraffic currently supports the following tasks:

* Traffic State Prediction
  * Traffic Flow Prediction
  * Traffic Speed Prediction
  * On-Demand Service Prediction
* Trajectory Next-Location Prediction

## Features

* **Unified**: LibTraffic builds a systematic pipeline to implement, use and evaluate traffic prediction models in a unified platform. We design basic spatial-temporal data storage, unified model instantiation interfaces, and standardized evaluation procedure.

* **Comprehensive**: 42 models covering four traffic prediction tasks have been reproduced to form a comprehensive model warehouse. Meanwhile, LibTraffic collects 29 commonly used datasets of different sources and implements a series of commonly used evaluation metrics and strategies for performance evaluation. 

* **Extensible**: LibTraffic enables a modular design of different components, allowing users to flexibly insert customized components into the library. Therefore, new researchers can easily develop new models with the support of LibTraffic.

## Overall Framework

![](https://bigscity-libtraffic-docs.readthedocs.io/en/latest/_images/framework.png)

* **Configuration Module**: Responsible for managing all the parameters involved in the framework.
* **Data Module**: Responsible for loading datasets and data preprocessing operations.
* **Model Module**: Responsible for initializing the reproduced baseline model or custom model.
* **Evaluation Module**: Responsible for evaluating model prediction results through multiple indicators.
* **Execution Module**: Responsible for model training and prediction.

## Installation

LibTraffic can only be installed from source code.

Please execute the following command to get the source code.

```shell
git clone https://github.com/LibTraffic/Bigscity-LibTraffic
cd Bigscity-LibTraffic
```

More details about environment configuration is represented in [Docs](https://bigscity-libtraffic-docs.readthedocs.io/en/latest/get_started/install.html).

## Quick-Start

Before run models in LibTraffic, please make sure you download at least one dataset and put it in directory `./raw_data/`. The dataset link is [BaiduDisk with code 1231](https://pan.baidu.com/s/1qEfcXBO-QwZfiT0G3IYMpQ) or [Google Drive](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing).

The script `run_model.py` is used for training and evaluating a single model in LibTraffic. When run the `run_model.py`, you must specify the following three parameters, namely **task, dataset and model**.  

For example:

```sh
python run_model.py --task traffic_state_pred --model GRU --dataset METR_LA
```

This script will run the GRU model on the METR_LA dataset for traffic state prediction task under the default configuration. 

More details is represented in [Docs](https://bigscity-libtraffic-docs.readthedocs.io/en/latest/get_started/quick_start.html).

## Contribution

The LibTraffic is mainly developed and maintained by Beihang Interest Group on SmartCity ([BIGSCITY](https://www.bigcity.ai/)). The core developers of this library are [@aptx1231](https://github.com/aptx1231) and [@WenMellors](https://github.com/WenMellors). 

Several co-developers have also participated in the reproduction of  the model, the list of contributions of which is presented in the [reproduction contribution list](./contribution_list.md).

If you encounter a bug or have any suggestion, please contact us by [raising an issue](https://github.com/LibTraffic/Bigscity-LibTraffic/issues).

## License

LibTraffic uses [Apache License 2.0](https://github.com/LibTraffic/Bigscity-LibTraffic/blob/master/LICENSE.txt).


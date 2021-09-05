class AbstractEvaluator(object):

    def __init__(self, config):
        raise NotImplementedError('evaluator not implemented')

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(dict): 输入数据
        """
        raise NotImplementedError('evaluator collect not implemented')

    def evaluate(self):
        """
        返回之前收集到的所有 batch 的评估结果
        """
        raise NotImplementedError('evaluator evaluate not implemented')

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        raise NotImplementedError('evaluator save_result not implemented')

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        raise NotImplementedError('evaluator clear not implemented')

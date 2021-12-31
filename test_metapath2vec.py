import os.path as osp

from numpy import mat
import torch
from libcity.data.dataset.metapath2vec_dataset import *
from model.road_representation.Metapath2Vec import *
from tensorboardX import SummaryWriter

dataset = Metapath2VecDataSet()
test_data = dataset.get_data()
data_feature = dataset.get_data_feature()
print('加载数据完成')

metapath = [
    ('road', 'r2r', 'road'),
    ('road', 'r2l', 'link'),
    ('link', 'l2l', 'link'),
    ('link', 'l2r', 'road'),
    ('road', 'r2r', 'road')
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'metapath':metapath,
    'embedding_dim':64,
    'walk_length':20,
    'context_size':7,
    'walks_per_node':5,
    'num_negative_samples':3,
    'sparse':True,
}

model = Metapath2vec(config=config,data_feature=dataset.get_data_feature()).to(device)
# 管道共享张量不在WINDOWS允许
loader = model.loader(batch_size=64, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}'))
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Acc: {acc:.4f}'))


@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('road', batch=test_data['road'].y_index)  # 调用forward返回embedding
    y = test_data['road'].y # 对应的标签

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                      max_iter=150)

if __name__ == '__main__':
    for epoch in range(1, 20):
        train(epoch)
        acc = test()
        print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
    writer = SummaryWriter('runs/embedding_example')
    writer.add_embedding(mat=model('road', batch=test_data['road'].y_index),metadata=test_data['road'].y)


# 加载数据完成
# Epoch: 1, Step: 00100/595, Loss: 3.7320
# Epoch: 1, Step: 00200/595, Loss: 3.0515
# Epoch: 1, Step: 00300/595, Loss: 2.5211
# Epoch: 1, Step: 00400/595, Loss: 2.1207
# Epoch: 1, Step: 00500/595, Loss: 1.8098
# Epoch: 1, Accuracy: 0.4728
# Epoch: 2, Step: 00100/595, Loss: 1.3735
# Epoch: 2, Step: 00200/595, Loss: 1.2330
# Epoch: 2, Step: 00300/595, Loss: 1.1266
# Epoch: 2, Step: 00400/595, Loss: 1.0456
# Epoch: 2, Step: 00500/595, Loss: 0.9848
# Epoch: 2, Accuracy: 0.4817
# Epoch: 3, Step: 00100/595, Loss: 0.8849
# Epoch: 3, Step: 00200/595, Loss: 0.8581
# Epoch: 3, Step: 00300/595, Loss: 0.8394
# Epoch: 3, Step: 00400/595, Loss: 0.8249
# Epoch: 3, Step: 00500/595, Loss: 0.8142
# Epoch: 3, Accuracy: 0.4789
# Epoch: 4, Step: 00100/595, Loss: 0.7896
# Epoch: 4, Step: 00200/595, Loss: 0.7870
# Epoch: 4, Step: 00300/595, Loss: 0.7852
# Epoch: 4, Step: 00400/595, Loss: 0.7845
# Epoch: 4, Step: 00500/595, Loss: 0.7822
# Epoch: 4, Accuracy: 0.4866
# Epoch: 5, Step: 00100/595, Loss: 0.7690
# Epoch: 5, Step: 00200/595, Loss: 0.7692
# Epoch: 5, Step: 00300/595, Loss: 0.7672
# Epoch: 5, Step: 00400/595, Loss: 0.7646
# Epoch: 5, Step: 00500/595, Loss: 0.7614
# Epoch: 5, Accuracy: 0.4873
# Epoch: 6, Step: 00100/595, Loss: 0.7483
# Epoch: 6, Step: 00200/595, Loss: 0.7465
# Epoch: 6, Step: 00300/595, Loss: 0.7458
# Epoch: 6, Step: 00400/595, Loss: 0.7437
# Epoch: 6, Step: 00500/595, Loss: 0.7422
# Epoch: 6, Accuracy: 0.4842
# Epoch: 7, Step: 00100/595, Loss: 0.7330
# Epoch: 7, Step: 00200/595, Loss: 0.7328
# Epoch: 7, Step: 00300/595, Loss: 0.7320
# Epoch: 7, Step: 00400/595, Loss: 0.7312
# Epoch: 7, Step: 00500/595, Loss: 0.7300
# Epoch: 7, Accuracy: 0.4899
# Epoch: 8, Step: 00100/595, Loss: 0.7251
# Epoch: 8, Step: 00200/595, Loss: 0.7250
# Epoch: 8, Step: 00300/595, Loss: 0.7250
# Epoch: 8, Step: 00400/595, Loss: 0.7249
# Epoch: 8, Step: 00500/595, Loss: 0.7244
# Epoch: 8, Accuracy: 0.4944
# Epoch: 9, Step: 00100/595, Loss: 0.7206
# Epoch: 9, Step: 00200/595, Loss: 0.7210
# Epoch: 9, Step: 00300/595, Loss: 0.7211
# Epoch: 9, Step: 00400/595, Loss: 0.7208
# Epoch: 9, Step: 00500/595, Loss: 0.7210
# Epoch: 9, Accuracy: 0.4917
# Epoch: 10, Step: 00100/595, Loss: 0.7180
# Epoch: 10, Step: 00200/595, Loss: 0.7186
# Epoch: 10, Step: 00300/595, Loss: 0.7190
# Epoch: 10, Step: 00400/595, Loss: 0.7193
# Epoch: 10, Step: 00500/595, Loss: 0.7192
# Epoch: 10, Accuracy: 0.4948
# Epoch: 11, Step: 00100/595, Loss: 0.7165
# Epoch: 11, Step: 00200/595, Loss: 0.7172
# Epoch: 11, Step: 00300/595, Loss: 0.7180
# Epoch: 11, Step: 00400/595, Loss: 0.7181
# Epoch: 11, Step: 00500/595, Loss: 0.7180
# Epoch: 11, Accuracy: 0.4979
# Epoch: 12, Step: 00100/595, Loss: 0.7155
# Epoch: 12, Step: 00200/595, Loss: 0.7159
# Epoch: 12, Step: 00300/595, Loss: 0.7165
# Epoch: 12, Step: 00400/595, Loss: 0.7162
# Epoch: 12, Step: 00500/595, Loss: 0.7165
# Epoch: 12, Accuracy: 0.5001
# Epoch: 13, Step: 00100/595, Loss: 0.7141
# Epoch: 13, Step: 00200/595, Loss: 0.7143
# Epoch: 13, Step: 00300/595, Loss: 0.7149
# Epoch: 13, Step: 00400/595, Loss: 0.7145
# Epoch: 13, Step: 00500/595, Loss: 0.7144
# Epoch: 13, Accuracy: 0.5007
# Epoch: 14, Step: 00100/595, Loss: 0.7124
# Epoch: 14, Step: 00200/595, Loss: 0.7127
# Epoch: 14, Step: 00300/595, Loss: 0.7128
# Epoch: 14, Step: 00400/595, Loss: 0.7130
# Epoch: 14, Step: 00500/595, Loss: 0.7128
# Epoch: 14, Accuracy: 0.4967
# Epoch: 15, Step: 00100/595, Loss: 0.7109
# Epoch: 15, Step: 00200/595, Loss: 0.7110
# Epoch: 15, Step: 00300/595, Loss: 0.7115
# Epoch: 15, Step: 00400/595, Loss: 0.7115
# Epoch: 15, Step: 00500/595, Loss: 0.7113
# Epoch: 15, Accuracy: 0.4978
# Epoch: 16, Step: 00100/595, Loss: 0.7097
# Epoch: 16, Step: 00200/595, Loss: 0.7098
# Epoch: 16, Step: 00300/595, Loss: 0.7102
# Epoch: 16, Step: 00400/595, Loss: 0.7097
# Epoch: 16, Step: 00500/595, Loss: 0.7100
# Epoch: 16, Accuracy: 0.4924
# Epoch: 17, Step: 00100/595, Loss: 0.7084
# Epoch: 17, Step: 00200/595, Loss: 0.7086
# Epoch: 17, Step: 00300/595, Loss: 0.7087
# Epoch: 17, Step: 00400/595, Loss: 0.7086
# Epoch: 17, Step: 00500/595, Loss: 0.7085
# Epoch: 17, Accuracy: 0.4985
# Epoch: 18, Step: 00100/595, Loss: 0.7075
# Epoch: 18, Step: 00200/595, Loss: 0.7076
# Epoch: 18, Step: 00300/595, Loss: 0.7077
# Epoch: 18, Step: 00400/595, Loss: 0.7077
# Epoch: 18, Step: 00500/595, Loss: 0.7077
# Epoch: 18, Accuracy: 0.4925
# Epoch: 19, Step: 00100/595, Loss: 0.7068
# Epoch: 19, Step: 00200/595, Loss: 0.7067
# Epoch: 19, Step: 00300/595, Loss: 0.7068
# Epoch: 19, Step: 00400/595, Loss: 0.7070
# Epoch: 19, Step: 00500/595, Loss: 0.7070
# Epoch: 19, Accuracy: 0.4962
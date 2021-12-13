# Layer List
| Layer | Model |  Test |
| :- | :-: |  :-:|
|Spatial.gnn.GCN|-| √ | 
|Spatial.gnn.GAT|-|  × |
|Spatial.gnn.AVWGCN|AGCRN|√|
|Spatial.gnn.LearnedGCN|STAGGCN|×|
|Spatial.gnn.ChebConvWithSAt|ASTGCN|√|
|Spatial.cnn.CNN|-|  √ | 
|Spatial.cnn.SpatialViewCNN|ASTGCN| √ | 
|Spatial.atten.SpatialAttention|GMAN|  √ | 
|Spatial.embedding.Embed|GMAN| √ | 
|Spatial.embedding.PositionalEmbedding|DCRNN| × | 
|Spatial.embedding.STEmbedding |GMAN| √  |
|Spatial.embedding.MultiEmbed|DeepMove|  × |
|Temporal.rnn.TGRU|-| √  |
|Temporal.rnn.TGCGRU|TGCN|  √  |
|Temporal.rnn.AGCGRU|AGCRN|  √  |
|Temporal.rnn.TLSTM|-|  √  |
|Temporal.cnn.TemporalCNN|-|  √  |
|Temporal.tcn.TemporalConvNet|TCN|  ×  |
|Temporal.atten.TemporalAttention|GMAN|  √  |
|output.atten.SelfAttention|-|  √  |
|output.atten.TransformAttention|GMAN|  √  |
|output.encoder_decoder.GMANEncoder|GMAN| √  |
|output.encoder_decoder.GMANDecoder|GMAN| √  |
|output.mlp.FusionLayer|ASTGCN| √  |
|output.mlp.FC|GMAN| √  |
# Todo List
| 计划-12.13 |
| :-： |
|调整代码结构|
|补充GNN网络和WaveNet|
|实现Config类，可以覆盖Spatial下的所有Class|
|测试代码|

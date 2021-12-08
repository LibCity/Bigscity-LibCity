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
|output.atten.TransformAttention|GMAN|  √  |
|output.encoder_decoder.GMANEncoder|GMAN| √  |
|output.encoder_decoder.GMANDecoder|GMAN| √  |
|output.mlp.FusionLayer|ASTGCN| √  |
|output.mlp.FC|GMAN| √  |
# Todo List
| Layer | Model |
| :- | :-: |
|output.mlp.FC(example)|GMAN|


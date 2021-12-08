# Layer List
| Layer | Model | Completer | Test |
| :- | :-: | :-: | :-:|
|Spatial.gnn.GCN|-| ZWT | √ | 
|Spatial.gnn.GAT|-| ZWT | × |
|Spatial.gnn.AVWGCN|AGCRN|ZWT|√|
|Spatial.gnn.LearnedGCN|STAGGCN|ZWT|×|
|Spatial.gnn.ChebConvWithSAt|ASTGCN|ZWT|√|
|Spatial.cnn.CNN|-| ZWT | √ | 
|Spatial.cnn.SpatialViewCNN|ASTGCN| ZWT | √ | 
|Spatial.atten.SpatialAttention|GMAN| ZWT | √ | 
|Spatial.embedding.Embed|GMAN| ZWT | √ | 
|Spatial.embedding.PositionalEmbedding|DCRNN| ZWT | × | 
|Spatial.embedding.STEmbedding |GMAN| ZWT | √  |
|Spatial.embedding.MultiEmbed|DeepMove| ZWT | × |
|Temporal.rnn.TGRU|-| ZWT | √  |
|Temporal.rnn.TGCGRU|TGCN| ZWT | √  |
|Temporal.rnn.AGCGRU|AGCRN| ZWT | √  |
|Temporal.rnn.TLSTM|-| ZWT | √  |
|Temporal.cnn.TemporalCNN|-| ZWT | √  |
|Temporal.tcn.TemporalConvNet|TCN| ZWT | ×  |
|Temporal.atten.TemporalAttention|GMAN| ZWT | √  |
|output.atten.TransformAttention|GMAN| ZWT | √  |
|output.encoder_decoder.GMANEncoder|GMAN| ZWT | √  |
|output.encoder_decoder.GMANDecoder|GMAN| ZWT | √  |
|output.mlp.FusionLayer|ASTGCN| ZWT | √  |
|output.mlp.FC|GMAN| ZWT | √  |
# Todo List
| Layer | Model |
| :- | :-: |
|output.mlp.FC(example)|GMAN|


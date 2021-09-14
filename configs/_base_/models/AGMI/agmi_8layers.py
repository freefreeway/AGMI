model = dict(
    type='AGMIDRPNet',
    drper=dict(
        type='AGMIDRPer',
        drug_encoder=dict(
            type='DrugGINEncoder',
            drug_features=78,
            output_dim=128,
            dropout=0.2,
            hidden_dim=32
        ),
        genes_encoder=dict(
            type='AGMIEncoder',
            out_channels=3,
            num_layers=8,
            aggr='add',
            bias=True,
        ),
        head=dict(
            type='BaseFusionHead',
            d_in_channels=128,
            g_in_channels=128,
            out_channels=1,
            reduction=2,
            dropout=0.2
        ),
        neck=dict(
            type='Conv1dNeck',
            in_channels=3,
            out_channels=128,
            kernel_size=16,
            pooling_size=6,
            dropout=0.2
        ),
    ),
    loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
)

train_cfg = None
test_cfg = dict(metrics=['MAE', 'MSE', 'RMSE',
                         'R2', 'PEARSON', 'SPEARMAN'])

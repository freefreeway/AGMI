exp_name = 'me_non_blind_1fold_8layers'

_base_ = [
    '../_base_/models/AGMI/agmi_8layers.py',
    '../_base_/drp_dataset/drugs_genes_dataset.py',
    '../_base_/exp_setting/base_setting.py'
]

model = dict(
    drper=dict(
        genes_encoder=dict(
            num_layers=8,
        ),
    ),
)

data = dict(
    train=dict(
        data_items='./data/non_blind_1_fold_tr_items.npy',
        name='MultiEdgeGraphGenesV5_1fold_tr',
        include_omic=['expr', 'mut', 'cn']
    ),
    val=dict(
        data_items='./data/non_blind_1_fold_val_items.npy',
        name='MultiEdgeGraphGenesV5_1fold_val',
        include_omic=['expr', 'mut', 'cn']
    ),
)

custom_hooks = [
    dict(type='TensorboardXHook',
         priority=85,
         log_dir='./work_dir/genes_drug_data/result/',
         interval=5000,
         exp_name=exp_name,
         ignore_last=True,
         reset_flag=False,
         by_epoch=False
         ),
    dict(type='MEHook',
         priority='VERY_LOW',
         gsea_path='./data/GSEA_edge_indexes_all_pairs_676_weighted.npy',
         ppi_path='./data/STRING_edge_index_all_10463182_pairs_weighted.npy',
         pearson_path='./data/edge_index_pearson_0.6.npy',
         num_nodes=18498
         )
]

work_dir = f'./work_dir/genes_drug_data/result/{exp_name}'

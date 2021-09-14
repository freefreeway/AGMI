# optimizer
optimizers = dict(drper=dict(type='Adam', lr=1e-4))

# learning policy
total_iters = 1000000
lr_config = dict(policy='Step', by_epoch=False, step=[200000, 400000, 800000, 1000000], gamma=0.5)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False, max_keep_ckpts=10)
evaluation = dict(interval=5000)
visual_config = None

# runtime settings
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
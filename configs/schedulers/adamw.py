paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        ".cls_token": dict(decay_mult=0.0),
    },
)

optimizer = dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg,
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="CosineAnnealing", min_lr=0, by_epoch=False)
runner = dict(type="EpochBasedRunner", max_epochs=5)

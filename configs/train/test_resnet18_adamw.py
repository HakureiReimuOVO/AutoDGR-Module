_base_ = [
    "../datasets/test.py",
    "../models/resnet18.py",
    "../schedulers/adamw.py",
    "../default_runtime.py",
]

model = dict(
    head=dict(
      num_classes={{_base_.dataset_num_classes}},
    )
)

model = dict(
    type="ImageClassifier",
    train_cfg=dict(model_name="resnet18"),
    backbone=dict(type="TIMMBackbone", model_name="resnet18", pretrained=True),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=1000,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1,),
    ),
)

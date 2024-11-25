model = dict(
    type="ImageClassifier",
    train_cfg=dict(model_name="resnet34"),
    backbone=dict(
        type="ResNet", depth=34, num_stages=4, out_indices=(3,), style="pytorch"
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=1000,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1, 5),
    ),
)


# if __name__ == '__main__':
#     from mmcls.models import build_classifier
#     built_model = build_classifier(model)

#     top_level_layers = set()
#     for name, module in built_model.backbone.named_modules():
#         parts = name.split('.')
#         if len(parts) == 2:
#             top_level_layers.add(name)

#     model_blocks = sorted(top_level_layers)
#     print(f'"resnet34": {model_blocks}')

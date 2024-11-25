checkpoint_config = dict(interval=5)

log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
evaluation = dict(interval=1, metric="accuracy")

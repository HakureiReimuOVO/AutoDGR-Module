import os
import itertools

dataset_folder = "configs/datasets"
model_folder = "configs/models"
scheduler_folder = "configs/schedulers"
default_runtime_file = "configs/default_runtime.py"
output_dir = "configs/train"

if __name__ == "__main__":
    dataset_files = list(os.listdir(dataset_folder))
    model_files = list(os.listdir(model_folder))
    scheduler_files = list(os.listdir(scheduler_folder))
    combinations = list(itertools.product(dataset_files, model_files, scheduler_files))

    #     for dataset, model, scheduler in combinations:
    #         model_file_path = os.path.join(model_folder, model)
    #         with open(model_file_path, "r") as model_file:
    #             model_content = model_file.read()
    #         config = f"""_base_ = [
    #     "../datasets/{dataset}",
    #     "../schedulers/{scheduler}",
    #     "../default_runtime.py",
    # ]

    # {model_content}
    # """

    for dataset, model, scheduler in combinations:
        config = f"""_base_ = [
    "../datasets/{dataset}",
    "../models/{model}",
    "../schedulers/{scheduler}",
    "../default_runtime.py",
]

model = dict(
    head=dict(
      num_classes={{{{_base_.dataset_num_classes}}}},
    )
)
"""
        dataset_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = os.path.splitext(os.path.basename(model))[0]
        scheduler_name = os.path.splitext(os.path.basename(scheduler))[0]
        filename = f"{dataset_name}_{model_name}_{scheduler_name}.py"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            f.write(config)

        print(f"Config written to {filepath}")

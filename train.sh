export LC_ALL=en_US.UTF-8
export PYTHONPATH="$PWD"

for file in configs/train/*.py; do
    echo "Training $file"
    python tools/train.py "$file" --device cpu
done

# python tools/train.py configs/train/test_resnet18_adamw.py --device cpu

# python tools/train.py configs/train/test_resnet18-pretrained_adamw.py --device cpu

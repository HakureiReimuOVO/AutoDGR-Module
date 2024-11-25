export LC_ALL=en_US.UTF-8
export PYTHONPATH="$PWD"

# for file in configs/datasets/*.py; do
#     python graph/dataset_graph.py "$file"
# done

python graph/build_dataset_graph.py configs/datasets data/dataset_graph.pkl

# python graph/update_dataset_graph.py configs/datasets/ttestt.py data/dataset_graph.pkl data/new_dataset_graph.pkl
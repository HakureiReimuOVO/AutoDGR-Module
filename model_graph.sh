export LC_ALL=en_US.UTF-8
export PYTHONPATH="$PWD"

python graph/build_model_graph.py configs/models configs/datasets/test.py data/model_graph.pkl

# python graph/update_model_graph.py configs/models/resnet.py configs/datasets/test.py data/new_model_graph.pkl data/new_model_graph.pkl
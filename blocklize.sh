export LC_ALL=en_US.UTF-8
export PYTHONPATH="$PWD"


# STEP 1
# for config_file in configs/train/*.py; do
#   python block/get_rep.py "$config_file" --out "result/feat_out/$(basename "$config_file" .py).pth"
# done

# STEP 2
# python block/compute_sim.py --feat_path result/feat_out --out result/sim_out --sim_func cka

# STEP 3
# python block/count_inout_size.py --feat_path result/feat_out --out result/inout_size

# STEP 4
python block/partition.py --out result/block_out --sim_path result/sim_out \
    --K 4 \
    --trial 50 \
    --eps 0.25 \
    --num_iter 50
net_type="MLP"
save_dir="../D3QN_norm"
alpha=10
gamma=0.95
# epsilon=0.05
post_process="False"
instance_time_limit=3600
n_min=150
n_max=200

for epsilon_test in 0.9 0; do
    D:/ruanjian/Anaconda/envs/pytorch/python DQN_test.py \
    --net_type $net_type --save_dir $save_dir --post_process $post_process\
    --alpha $alpha --epsilon $epsilon_test \
    --instance_time_limit $instance_time_limit --n_min $n_min --n_max $n_max

    sleep 1
done

# mv $save_dir/*.pkl $save_dir/models

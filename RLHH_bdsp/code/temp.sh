net_type="MLP"
save_dir="../D3QN_norm"
num_episodes=600
batch_size=128
buffer_size=100000
alpha=10
gamma=0.95
# epsilon=0.05
post_process="False"
instance_time_limit=600
n_min=50
n_max=100

for epsilon_test in 0.9 0; do
    D:/ruanjian/Anaconda/envs/pytorch/python DQN_test.py \
    --net_type $net_type --save_dir $save_dir --post_process $post_process\
    --alpha $alpha --epsilon $epsilon_test \
    --instance_time_limit $instance_time_limit --n_min $n_min --n_max $n_max

    sleep 1
done

# mv $save_dir/*.pkl $save_dir/models

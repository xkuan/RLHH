net_type="QNetTwinDuel"
save_dir="../D3QN"
alpha=100
test_cases=(
#    '600 small'
#    '3600 large'
    '3600 verylarge'
)

for i in "${test_cases[@]}"; do
    case=($i)
    instance_time_limit=${case[0]}
    instance_size=${case[1]}

    echo $instance_time_limit
    echo $instance_size

    D:/ruanjian/Anaconda/envs/pytorch/python DQN_test.py \
    --net_type $net_type --save_dir $save_dir --alpha ${alpha} \
    --instance_time_limit $instance_time_limit --instance_size $instance_size
    
    sleep 1
done

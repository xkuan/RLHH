net_type="MLP"
save_dir="../D3QN/one"
run_mode="train"
num_episodes=1000
batch_size=64
buffer_size=100000
alpha=100
epsilon=0.05

for gamma in 0.95; do
	for lr in 0.1 0.01; do
		for seed in 123 523; do
			D:/ruanjian/Anaconda/envs/pytorch/python D3QN_train.py \
			--net_type $net_type --save_dir $save_dir --run_mode $run_mode \
			--num_episodes $num_episodes --batch_size $batch_size --buffer_size $buffer_size \
			--alpha $alpha --epsilon $epsilon --gamma $gamma --learning_rate $lr --seed $seed
			sleep 1
		done
	done
done

# test_cases=(
#     '600 small'
#     # '3600 large_r'
# )

# for i in "${test_cases[@]}"; do
#     case=($i)
#     instance_time_limit=${case[0]}
#     instance_size=${case[1]}

#     echo $instance_time_limit
#     echo $instance_size

#     D:/ruanjian/Anaconda/envs/pytorch/python DQN_test.py \
#     --net_type $net_type --save_dir $save_dir --alpha ${alpha} \
#     --instance_time_limit $instance_time_limit --instance_size $instance_size
    
#     sleep 1
# done


# num_episodes=2000
# batch_size=64
# buffer_size=100000
# load_model=True

# good_cases=(
#     '10 0.01 0.9 0.1 123'
#     '10 0.01 0.9 0.01 523'
#     '10 0.01 0.95 0.1 123'
#     '10 0.01 0.95 0.1 523'
#     '10 0.01 0.95 0.01 123'
#     '10 0.01 0.95 0.01 523'
#     '10 0.05 0.95 0.1 123'
#     '10 0.05 0.95 0.1 523'
#     '10 0.05 0.95 0.01 123'
#     '10 0.05 0.95 0.01 523'
#     '100 0.01 0.9 0.01 523'
#     '100 0.01 0.95 0.1 123'
#     '100 0.01 0.95 0.1 523'
#     '100 0.01 0.95 0.01 123'
#     '100 0.05 0.95 0.1 123'
#     '100 0.05 0.95 0.1 523'
#     '100 0.05 0.95 0.01 123'
#     '100 0.05 0.95 0.01 523'
# )

# for i in "${good_cases[@]}"; do
#     case=($i)
#     alpha=${case[0]}
#     epsilon=${case[1]}  # 0.05
#     gamma=${case[2]}    # 0.95
#     lr=${case[3]}
#     seed=${case[4]}
#     load_case=a="$alpha"_e="$epsilon"_g="$gamma"_lr="$lr"_seed="$seed"
#     echo $load_case
#     D:/ruanjian/Anaconda/envs/pytorch/python DQN_train.py \
#     --num_episodes ${num_episodes} --batch_size ${batch_size} --buffer_size ${buffer_size} \
#     --alpha ${alpha} --epsilon ${epsilon} --gamma ${gamma} --learning_rate ${lr} --seed ${seed} \
#     --load_model ${load_model} --load_case ${load_case}

#     sleep 1
# done


exit

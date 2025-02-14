net_type="MLP"
save_dir="../D3QN_norm"
num_episodes=600
batch_size=128
buffer_size=100000
# alpha=100
gamma=0.95
# epsilon=0.05
post_process="False"

for alpha in 10; do
    for epsilon in 0.1; do
        for lr in 0.05 0.01; do
            for seed in 123 523 678; do

                run_mode="train"
                D:/ruanjian/Anaconda/envs/pytorch/python D3QN_train.py \
                --net_type $net_type --save_dir $save_dir --run_mode $run_mode \
                --num_episodes $num_episodes --batch_size $batch_size --buffer_size $buffer_size \
                --gamma $gamma --post_process $post_process\
                --alpha $alpha --epsilon $epsilon --learning_rate $lr --seed $seed
                
                sleep 1

                instance_time_limit=600
                n_min=50
                n_max=50
                for epsilon_test in 0 0.5; do
                    D:/ruanjian/Anaconda/envs/pytorch/python DQN_test.py \
                    --net_type $net_type --save_dir $save_dir --post_process $post_process\
                    --alpha $alpha --epsilon $epsilon_test \
                    --instance_time_limit $instance_time_limit --n_min $n_min --n_max $n_max
                
                    sleep 1
                done

                # mv $save_dir/*.pkl $save_dir/models

            done
        done
    done
done
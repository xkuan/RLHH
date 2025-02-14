num_episodes=20000
batch_size=64
buffer_size=50000
load_model=True

good_cases=(
    '10 0.01 0.9 0.1 123'
    '10 0.01 0.9 0.01 523'
    '10 0.01 0.95 0.1 123'
    '10 0.01 0.95 0.1 523'
    '10 0.01 0.95 0.01 123'
    '10 0.01 0.95 0.01 523'
    '10 0.05 0.95 0.1 123'
    '10 0.05 0.95 0.1 523'
    '10 0.05 0.95 0.01 123'
    '10 0.05 0.95 0.01 523'
    '100 0.01 0.9 0.01 523'
    '100 0.01 0.95 0.1 123'
    '100 0.01 0.95 0.1 523'
    '100 0.01 0.95 0.01 123'
    '100 0.05 0.95 0.1 123'
    '100 0.05 0.95 0.1 523'
    '100 0.05 0.95 0.01 123'
    '100 0.05 0.95 0.01 523'
)

# good_cases[1]=(10 0.01)
# good_cases[2]=(100 0.05)

for i in "${good_cases[@]}"; do
    case=($i)
    alpha=${case[0]}
    epsilon=${case[1]}  # 
    gamma=${case[2]}    # 0.95
    lr=${case[3]}
    seed=${case[4]}
    load_dir=a="$alpha"_e="$epsilon"_g="$gamma"_lr="$lr"_seed="$seed"
    echo $load_dir
    # D:/ruanjian/Anaconda/envs/pytorch/python DQN_train.py \
    # --num_episodes ${num_episodes} --batch_size ${batch_size} --buffer_size ${buffer_size} \
    # --alpha ${alpha} --epsilon ${epsilon} --gamma ${gamma} --learning_rate ${lr} --seed ${seed} \
    # --load_model ${load_model} --load_dir ${load_dir}

    sleep 1
done

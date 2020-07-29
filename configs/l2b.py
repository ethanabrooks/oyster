# default PEARL experiment settings
# all experiments should modify these settings only as needed
l2b_config = dict(
    env_name="l2b",
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5,  # dimension of the latent context vector
    net_size=300,  # number of units per FC layer in each network
    path_to_weights=None,  # path to pre-trained weights to load into networks
    env_params={
        "actor_freq": 1,
        "batch_size": 64,
        "buffer_size": 2000000,
        "discount": 0.99,
        "env_id": "levels",
        "eval_episodes": 10,
        "eval_freq": 5000,
        "learning_rate": 0.005,
        "load_path": None,
        "max_time_steps": 1000000,
        "render": False,
        "save_freq": 5000,
        "save_model": False,
        "seed": 5,
        "start_time_steps": 10000,
        "tau": 0.05,
        "train_steps": 1,
    },
    algo_params=dict(
        meta_batch=16,  # number of tasks to average the gradient across
        num_iterations=500,  # number of data sampling / training iterates
        num_initial_steps=2000,  # number of transitions collected per task before training
        num_tasks_sample=5,  # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400,  # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0,  # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400,  # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000,  # number of meta-gradient steps taken per iteration
        num_evals=2,  # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256,  # number of transitions in the RL batch
        embedding_batch_size=64,  # number of transitions in the context batch
        embedding_mini_batch_size=64,  # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200,  # max path length for this environment
        discount=0.99,  # RL discount factor
        soft_target_tau=0.005,  # for SAC target network update
        policy_lr=3e-4,
        qf_lr=3e-4,
        vf_lr=3e-4,
        context_lr=3e-4,
        reward_scale=5.0,  # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False,  # whether to sparsify rewards as determined in env
        kl_lambda=0.1,  # weight on KL divergence term in encoder loss
        use_information_bottleneck=True,  # False makes latent context deterministic
        use_next_obs_in_context=False,  # use next obs if it is useful in distinguishing tasks
        update_post_train=1,  # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1,  # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False,  # recurrent or permutation-invariant encoder
        dump_eval_paths=False,  # whether to save evaluation trajectories
    ),
    util_params=dict(
        base_log_dir="output",
        use_gpu=True,
        gpu_id=0,
        debug=False,  # debugging triggers printing and writes logs to debug directory
        docker=False,  # TODO docker is not yet supported
    ),
)

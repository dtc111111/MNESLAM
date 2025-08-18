import os
import sys
import argparse
import config
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def spawn_agent(rank, world_size, config_path, shared_components):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # 每个智能体加载自己的配置
    agent_cfg = config.load_config(config_path)
    # 将 agent_id 添加到配置中，用于后续数据加载
    agent_cfg['agent_id'] = rank
    # if args.output is not None:
    #     agent_cfg['data']['output'] = args.output

    from mneslam_mp import run_agent

    run_agent(rank, world_size, agent_cfg, shared_components)

if __name__ == '__main__':

    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running MNE-SLAM, single or multi-agent.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for multi-agent simulation.')
    
    args = parser.parse_args()

    world_size = args.num_gpus

    # Create shared data structures for all agents
    manager = mp.Manager()
    shared_components = {
        'descriptor_db': manager.list(),      # 共享的NetVLAD描述子数据库
        'descriptor_db_lock': manager.Lock()  # 描述子数据库的锁
    }

    if world_size > 1:
        print(f"Spawning {world_size} agents...")
        base_config_path = args.config

        config_base_name = base_config_path.rsplit('.yaml', 1)[0]

        processes = []
        for rank in range(world_size):
            # 为每个智能体构建特定的配置文件路径
            agent_config_path = f"{config_base_name}_agent{rank}.yaml"
            print(f"Launching agent {rank} with config: {agent_config_path}")
            p = mp.Process(target=spawn_agent, args=(rank, world_size, agent_config_path, shared_components))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print("Running in single agent mode...")
        cfg = config.load_config(args.config)
        if args.output is not None:
            cfg['data']['output'] = args.output
        # For a single agent, we don't need to spawn a new process
        from mneslam_mp import MNESLAM
        slam = MNESLAM(cfg)
        slam.run()
        slam.terminate(rank=-1)
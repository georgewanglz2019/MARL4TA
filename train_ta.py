import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import json

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv

"""Train script for MPEs."""


def make_train_env(all_args):
    if all_args.transportation_net == 'siouxfalls':
        from TAEnvs.load_net_multi_agent_config import load_SiouxFalls_so_action_set_multi_agent_config  # load siouxfalls
        from TAEnvs.STA_Continuous_MultiAgent_Env_V1 import STA_Continuous_MultiAgent_Env_V1
        args_abs_path = os.path.join(os.getcwd(), 'TAEnvs', 'SiouxFallNetData')
        env_args = load_SiouxFalls_so_action_set_multi_agent_config(file_path=args_abs_path)

        env_args.ue_or_so = all_args.ue_or_so
        env_args.action_smoothing = all_args.action_smoothing
        env_args.max_env_iter_num = all_args.max_env_iter_num
        env_args.OW_dynamic_demand = all_args.OW_dynamic_demand

        print('+++++++++++++++++++++ Siouxfalls net is loaded +++++++++++++++++++++')

    elif all_args.transportation_net == 'Anaheim':
        from TAEnvs.load_net_multi_agent_config import load_Anaheim_so_action_set_multi_agent_config  # load Anaheim
        from TAEnvs.STA_Continuous_MultiAgent_Env_V1 import STA_Continuous_MultiAgent_Env_V1
        args_abs_path = os.path.join(os.getcwd(), 'TAEnvs', 'Anaheim')
        env_args = load_Anaheim_so_action_set_multi_agent_config(file_path=args_abs_path)

        env_args.ue_or_so = all_args.ue_or_so
        env_args.action_smoothing = all_args.action_smoothing
        env_args.max_env_iter_num = all_args.max_env_iter_num
        env_args.OW_dynamic_demand = all_args.OW_dynamic_demand

        print('+++++++++++++++++++++ Anaheim net is loaded +++++++++++++++++++++')

    elif all_args.transportation_net == 'OW':
        from TAEnvs.load_net_multi_agent_config import load_OW_ksp_multi_agent_config  # load Anaheim
        from TAEnvs.STA_Continuous_MultiAgent_Env_V1 import STA_Continuous_MultiAgent_Env_V1
        args_abs_path = os.path.join(os.getcwd(), 'TAEnvs', 'OW')
        env_args = load_OW_ksp_multi_agent_config(file_path=args_abs_path, ksp=6)

        env_args.ue_or_so = all_args.ue_or_so
        env_args.action_smoothing = all_args.action_smoothing
        env_args.max_env_iter_num = all_args.max_env_iter_num
        env_args.OW_dynamic_demand = all_args.OW_dynamic_demand

        print('+++++++++++++++++++++ Anaheim net is loaded +++++++++++++++++++++')

    else:
        raise ValueError(f'Net name = {all_args.transportation_net} not recognized')

    def get_env_fn(rank):
        def init_env():

            # from envs.env_continuous import ContinuousActionEnv
            # env = ContinuousActionEnv()

            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()

            env_sta = STA_Continuous_MultiAgent_Env_V1(args=env_args)
            env_sta.max_itertation_time = all_args.episode_length
            #print(env.action_space)

            from TAEnvs.env_continuous_sta import ContinuousActionEnv
            env = ContinuousActionEnv(env_base=env_sta)



            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    if all_args.transportation_net == 'siouxfalls':
        from TAEnvs.load_net_multi_agent_config import \
            load_SiouxFalls_so_action_set_multi_agent_config  # load siouxfalls
        from TAEnvs.STA_Continuous_MultiAgent_Env_V1 import STA_Continuous_MultiAgent_Env_V1
        args_abs_path = os.path.join(os.getcwd(), 'TAEnvs', 'SiouxFallNetData')
        env_args = load_SiouxFalls_so_action_set_multi_agent_config(file_path=args_abs_path)

        env_args.ue_or_so = all_args.ue_or_so
        env_args.action_smoothing = all_args.action_smoothing
        env_args.max_env_iter_num = all_args.max_env_iter_num
        env_args.OW_dynamic_demand = all_args.OW_dynamic_demand

        print('+++++++++++++++++++++ Siouxfalls net is loaded +++++++++++++++++++++')

    elif all_args.transportation_net == 'Anaheim':
        from TAEnvs.load_net_multi_agent_config import load_Anaheim_so_action_set_multi_agent_config  # load Anaheim
        from TAEnvs.STA_Continuous_MultiAgent_Env_V1 import STA_Continuous_MultiAgent_Env_V1
        args_abs_path = os.path.join(os.getcwd(), 'TAEnvs', 'Anaheim')
        env_args = load_Anaheim_so_action_set_multi_agent_config(file_path=args_abs_path)

        env_args.ue_or_so = all_args.ue_or_so
        env_args.action_smoothing = all_args.action_smoothing
        env_args.max_env_iter_num = all_args.max_env_iter_num
        env_args.OW_dynamic_demand = all_args.OW_dynamic_demand

        print('+++++++++++++++++++++ Anaheim net is loaded +++++++++++++++++++++')

    elif all_args.transportation_net == 'OW':
        from TAEnvs.load_net_multi_agent_config import load_OW_ksp_multi_agent_config  # load Anaheim
        from TAEnvs.STA_Continuous_MultiAgent_Env_V1 import STA_Continuous_MultiAgent_Env_V1
        args_abs_path = os.path.join(os.getcwd(), 'TAEnvs', 'OW')
        env_args = load_OW_ksp_multi_agent_config(file_path=args_abs_path, ksp=6)

        env_args.ue_or_so = all_args.ue_or_so
        env_args.action_smoothing = all_args.action_smoothing
        env_args.max_env_iter_num = all_args.max_env_iter_num
        env_args.OW_dynamic_demand = all_args.OW_dynamic_demand

        print('+++++++++++++++++++++ Anaheim net is loaded +++++++++++++++++++++')

    else:
        raise ValueError(f'Net name = {all_args.transportation_net} not recognized')

    def get_env_fn(rank):
        def init_env():

            env_sta = STA_Continuous_MultiAgent_Env_V1(args=env_args)
            env_sta.save_decison_details = True  # save mdp data
            env_sta.max_itertation_time = all_args.episode_length
            # print(env.action_space)

            from TAEnvs.env_continuous_sta import ContinuousActionEnv
            env = ContinuousActionEnv(env_base=env_sta)

            env.seed(all_args.seed + rank * 1000)

            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    #parser.add_argument("--num_agents", type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.num_agents
    print('num_agents=', num_agents)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    config_to_save = {
        "all_args": vars(all_args),
        "num_agents": num_agents,
        "device": str(device),
        "run_dir": str(run_dir)
    }

    # save config to json
    with open(str(run_dir / "train_config.json"), "w") as f:
        json.dump(config_to_save, f, indent=4)

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

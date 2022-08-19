import matplotlib.pyplot as plt
import pandas as pd
from plotting.utils import create_reward_dataframe_for_plotting
import os
from tb_results.paths import events_dict, local_events_dict

COLORS = ['#377eb8', '#4daf4a', '#984ea3', '#a65628', '#e41a1c']
NODE_INDEXES = ["NODE 97851", "NODE 71555", "NODE 109618"]


def generate_path(node_index, algo_name, events_list):
    name = f"{algo_name}_{node_index}"
    paths = []
    for i in range(len(events_list)):
        path = f"tb_results/{node_index}/{algo_name}/{name}_{i+1}_1/{events_list[i]}"
        paths.append(path)
    return name, paths

def generate_path_for_bigger_local_plots(node_index, algo_name, local_size, local_events_list):
    name = f"{algo_name}_{node_index}"
    paths = []
    for i in range(len(local_events_list)):
        path = f"tb_results/{node_index}/{algo_name}_local/local{local_size}/raidus_{local_size}_{i + 1}_1/{local_events_list[i]}"
        paths.append(path)
    return name, paths


def multi_plot(df_list, node_index, algo_names, save_dir):
    plt.clf()
    from matplotlib.pyplot import figure
    figure(figsize=(10, 7), dpi=80)
    for i in range(len(df_list)):
        df = df_list[i]
        steps = df['step']
        mean = df['mean']
        std = df['std']
        plt.plot(steps, mean, color=COLORS[i], label=algo_names[i])
        plt.fill_between(steps, mean - std, mean + std, alpha=0.3, color=COLORS[i])

    #plt.legend(bbox_to_anchor=(1.135, 1))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Mean Episode Reward', fontsize=15)
    # plt.title(f'NODE {node_index}')
    plt.savefig(os.path.join(save_dir + '/plots', f'{node_index}_reward_plot.png'))


def single_plot(df, name, save_dir):
    plt.clf()
    plt.rc('axes', axisbelow=True)
    steps = df['step']
    mean = df['mean']
    std = df['std']
    plt.plot(steps, mean)
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.xlabel('step')
    plt.ylabel('ep_rew_mean')
    plt.title(f'{name} reward plot')
    plt.savefig(os.path.join(save_dir + '/plots', f'{name}_reward_plot.png'))


def main(node_index):
    save_dir = f'plots_and_stats/{node_index}'
    algo_names = ['A2C', 'DDPG', 'PPO', 'TD3', 'TRPO']
    # algo_names = ['TRPO']

    df_list = []
    for i, algo_name in enumerate(algo_names):
        name, paths = generate_path(node_index, algo_name, events_dict[node_index][algo_name])
        df = create_reward_dataframe_for_plotting(paths)
        single_plot(df, name, save_dir)
        df_list.append(df)
        df.to_csv(os.path.join(save_dir + '/stats', f'stats_{name}.csv'))  # save as csv

    multi_plot(df_list, node_index, algo_names, save_dir)


def multi_plot_local(df_list, node_index, local_sizes, save_dir):
    plt.clf()
    plt.rc('axes', axisbelow=True)
    from matplotlib.pyplot import figure
    figure(figsize=(10, 7), dpi=80)
    for i in range(len(df_list)):
        df = df_list[i][0:245]
        steps = df['step']
        mean = df['mean']
        std = df['std']
        plt.plot(steps, mean, color=COLORS[i], label=f'L = {local_sizes[i]}')
        plt.fill_between(steps, mean - std, mean + std, alpha=0.3, color=COLORS[i])

    plt.legend()
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Mean Episode Reward', fontsize=15)
    #plt.title(f'NODE {node_index}')
    plt.savefig(os.path.join(save_dir + '/plots', f'{node_index}_local_size_reward_plot.png'))

def main_local(node_index):
    save_dir = f'plots_and_stats/{node_index}'
    local_sizes = [100, 250, 500]
    algo_name = 'PPO'

    df_list = []
    for i, local_size in enumerate(local_sizes):
        name, paths = generate_path_for_bigger_local_plots(node_index, algo_name, local_size, local_events_dict[node_index][local_size])
        df = create_reward_dataframe_for_plotting(paths)
        df_list.append(df)

    multi_plot_local(df_list, node_index, local_sizes, save_dir)


if __name__ == "__main__":
    node_index = 97851
    main_local(node_index)


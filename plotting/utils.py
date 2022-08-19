import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def convert_event_to_dataframe(path):
    log_data = pd.DataFrame({"metric": [], "step": [], "value": []})
    event_accumulator = EventAccumulator(path, {"scalars": 0})
    event_accumulator.Reload()
    tags = event_accumulator.Tags()["scalars"]
    for tag in tags:
        event_list = event_accumulator.Scalars(tag)
        step = list(map(lambda x: x.step, event_list))
        values = list(map(lambda x: x.value, event_list))
        dict = {"metric": [tag] * len(step), "step": step, "value": values}
        df = pd.DataFrame(dict)
        log_data = pd.concat([log_data, df])

    return log_data


def create_reward_dataframe_for_plotting(paths):
    """
    This function receives paths to event files and returns a dataframe with columns :
        metric, step, run_1, run_2, ..., run_n, mean, std
    The value of metric is 'rollout/ep_rew_mean' and the value of run_i columns are the logged rewards.
    All runs logging configurations (steps) should be identical.
    """

    # create list of dataframes, containing ep_rew_mean of each event file
    ep_rew_mean_dataframes = []
    for path in paths:
        df = convert_event_to_dataframe(path)
        df = df[df['metric'] == 'rollout/ep_rew_mean']
        ep_rew_mean_dataframes.append(df)

    # check if all the dataframes has same steps
    # assert all(x['step'].equals(ep_rew_mean_dataframes[0]['step']) for x in ep_rew_mean_dataframes)

    # creating the final dataframe
    final_df = pd.DataFrame()
    final_df['metric'] = ep_rew_mean_dataframes[0]['metric']
    final_df['step'] = ep_rew_mean_dataframes[0]['step']

    # add value (reward) of each dataframe to the final dataframe
    col_name_list = []
    for i, df in enumerate(ep_rew_mean_dataframes):
        col_name = 'run_' + str(i)
        final_df[col_name] = df['value']
        col_name_list.append(col_name)

    # calculate & add mean and std of each row
    final_df['mean'] = final_df[col_name_list].mean(axis=1)
    final_df['std'] = final_df[col_name_list].std(axis=1)

    return final_df

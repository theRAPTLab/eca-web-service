import dataclasses
import logging
import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_config(args_class, external_config):
    keys = {f.name for f in dataclasses.fields(args_class)}
    inputs = {k: v for k, v in external_config.items() if k in keys}
    return args_class(**inputs)


def gen_buffer_tensor(df, device="cpu") -> dict:
    """
    creates tensor for state, action, reward, next_state, done and action-musk
    :return: returns a dictionary of five tensor sets
    """
    states = np.stack(df['state'])
    actions = np.array(df['action'])
    rewards = np.array(df['reward'])
    dones = np.array(df['done'])

    next_states = np.stack(df['state'][1:])
    next_states = np.vstack([next_states, np.zeros(next_states.shape[1])])
    idx = np.where(dones == True)
    next_states[idx] = np.zeros(next_states.shape[1])

    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    action_tensor = torch.tensor(actions, dtype=torch.int64, device=device)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_state_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
    done_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    ret_dict = {'state': state_tensor, 'action': action_tensor, 'reward': reward_tensor,
                'next_state': next_state_tensor, 'done': done_tensor}
    return ret_dict


def sample_buffer_tensor(buffer_tensor: dict, sample_size: int) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    sample transactions of given batch size in tensor format
    :return: five tensors that contains samples of state, action, reward, next_state, done
    """
    total_rows = buffer_tensor['state'].size()[0]
    if sample_size == -1:
        idx = np.array(range(total_rows))
    else:
        idx = np.random.choice(range(total_rows), sample_size)

    state, action, reward, next_state, done = (buffer_tensor['state'][idx],
                                               buffer_tensor['action'][idx],
                                               buffer_tensor['reward'][idx],
                                               buffer_tensor['next_state'][idx],
                                               buffer_tensor['done'][idx])

    return state, action, reward, next_state, done


def load_data(df_location: str, test_size: float = 0.2) -> (pd.DataFrame, pd.DataFrame):
    logger.info("-- loading data from {0} with test size {1} --".format(df_location, test_size))
    df = pd.read_pickle(df_location)
    a = (df.groupby(['action']).count() / len(df))[['step']]
    a.rename(columns={'step': 'action_prob'}, inplace=True)
    df = df.merge(a, on=['action'], how='left')

    d = df.loc[df['done']]
    episode_ids = d['episode'].tolist()
    nlgs = d['reward'].tolist()

    train_episode, test_episode = train_test_split(episode_ids, test_size=test_size, stratify=nlgs)

    train_df = df.loc[df['episode'].isin(train_episode)].reset_index(drop=True)
    test_df = df.loc[df['episode'].isin(test_episode)].reset_index(drop=True)

    # shuffle
    train_df = train_df.set_index("episode").loc[train_episode].reset_index()
    test_df = test_df.set_index("episode").loc[test_episode].reset_index()

    return train_df, test_df

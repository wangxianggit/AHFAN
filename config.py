from pathlib import Path

__all__ = ['project_path', 'dataset_config']

project_path = Path(__file__).parent


dataset_config = {
    'elliptic':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':64,
            'lr_f': 3e-5,
            'lr': 3e-5,
            'weight_decay': 1e-5,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
    'yelp':
        {
            'K': 2,
            'M': 5,
            'm': 3,
            'hidden_channels': 64,
            'out_channels': 64,
            'lr_f': 5e-4,
            'lr': 5e-4,
            'weight_decay': 5e-4,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
    'weibo':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':64,
            'lr_f': 5e-5,
            'lr': 5e-5,
            'weight_decay': 1e-5,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
    'quest':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':128,
            'lr_f': 5e-4,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },

}
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNet(Dataset):
    def __init__(self, config, classes=['chair'], split='train', objects_ids=None, mixed=True, angles=[0., 0., np.pi / 2]):
        shapenet = np.load(config['data_dir'])
        print('ShapeNet\'s being loaded...')
        self.objects_ids = objects_ids
        self.cloud = []
        self.targets = []
        for _class in classes:
            if objects_ids:
                X = [shapenet[_class][idx] for idx in objects_ids]
            else:
                X = shapenet[_class]

                if split == 'train':
                    X = X[:int(.85 * len(X))]
                elif split == 'valid':
                    X = X[int(.85 * len(X)):int(.90 * len(X))]
                elif split == 'test':
                    X = X[int(.90 * len(X)):]
                else:
                    raise ValueError(r'Split must be \'train\', \'valid\' or \'test\'')
            if mixed:
                for target, cloud in enumerate(X):
                    self.cloud.extend(cloud)
                    self.targets.extend([target] * len(cloud))

                self.cloud = np.array(self.cloud).reshape(-1, 3)
            else:
                self.cloud = X
                self.targets = [i for i in range(len(X))]

            self.rotation_matrix = self.get_rotation_matrix(angles)

            self.cloud = np.dot(self.cloud, self.rotation_matrix)
            print('Done\n')

    def __len__(self):
        return len(self.cloud)

    def __getitem__(self, idx):
        point, target = self.cloud[idx], self.targets[idx]
        point, target = torch.from_numpy(point), torch.tensor(target)
        return point, target

    def get_rotation_matrix(self, angles):
        x, y, z = angles
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(x), -np.sin(x)],
                       [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)],
                       [0, 1, 0],
                       [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0],
                       [np.sin(z), np.cos(z), 0],
                       [0, 0, 1]])
        return np.dot(Rx, np.dot(Ry, Rz))
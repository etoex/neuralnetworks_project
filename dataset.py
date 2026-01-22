import numpy as np
import torch
from torch.utils.data import Dataset


class LogMelSeqDataset(Dataset):
    """
    Класс датасета для обучения нейросети оконной (фреймовой) классификации.

    Один элемент датасета соответствует одному аудиофайлу.

    Для каждого файла хранятся:
    - X: массив признаков
    - y: массив меток, где каждая метка — класс звука для соответствующего окна
    - id: идентификатор файла
    """

    def __init__(self, npz_path="features_logmel_mfcc.npz"):
        """
        Инициализация датасета.

        Загружаем заранее подготовленные признаки и метки из npz-файла.
        """
        data = np.load(npz_path, allow_pickle=True)

        self.X = data["X"]
        self.y = data["y"]
        self.ids = data["ids"]

    def __len__(self):
        """
        Возвращает количество элементов в датасете
        (т.е. количество аудиофайлов).
        """
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Возвращает один элемент датасета по индексу.

        Преобразует numpy-массивы в torch.Tensor,т.к. PyTorch работает именно с тензорами.
        """
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        file_id = str(self.ids[idx])

        return x, y, file_id


def collate_pad(batch):
    """
    Функция для подготовки батча переменной длины.
    """
    xs, ys, ids = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Tmax = int(lengths.max().item())
    F = int(xs[0].shape[1])

    X_pad = torch.zeros(len(xs), Tmax, F, dtype=torch.float32)

    y_pad = torch.full((len(xs), Tmax), fill_value=-1, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.shape[0]
        X_pad[i, :T] = x
        y_pad[i, :T] = y

    return X_pad, y_pad, lengths, list(ids)

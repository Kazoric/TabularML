from torch.utils.data import DataLoader, Dataset
from data.preprocessing import preprocessing

class TabularDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = X_cat
        self.X_num = X_num
        self.y = y
    
    def __len__(self):
        return len(self.X_num)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        else:
            return self.X_cat[idx], self.X_num[idx]

def get_dataloader(config):
    X_cat_train, X_cat_valid, X_num_train, X_num_valid, y_train, y_valid, X_cat_test, X_num_test = preprocessing(config)

    train_dataset = TabularDataset(X_cat_train, X_num_train, y_train)
    valid_dataset = TabularDataset(X_cat_valid, X_num_valid, y_valid)
    test_dataset = TabularDataset(X_cat_test, X_num_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
from data import MVTecDataset

def get_dataloader(cls,
                   size,
                   vanilla):
    train_loader, test_loader = MVTecDataset(cls, size=size, vanilla=vanilla).get_dataloaders()
    return train_loader, test_loader
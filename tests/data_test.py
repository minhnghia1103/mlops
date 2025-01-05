import torch
from torch.utils.data import DataLoader

def test_trainset():
    # Thử tải trainset
    trainset = torch.load('./data/trainset.pt')
    assert trainset is not None, "Trainset is None."
    assert len(trainset) > 0, "Trainset is empty."

    # Kiểm tra DataLoader
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    for images, labels in trainloader:
        assert images.shape[1:] == (3, 32, 32), "Trainset images have incorrect shape."
        assert len(labels) > 0, "Trainset labels are empty."
        break

def test_testset():
    # Thử tải testset
    testset = torch.load('./data/testset.pt')
    assert testset is not None, "Testset is None."
    assert len(testset) > 0, "Testset is empty."

    # Kiểm tra DataLoader
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    for images, labels in testloader:
        assert images.shape[1:] == (3, 32, 32), "Testset images have incorrect shape."
        assert len(labels) > 0, "Testset labels are empty."
        break

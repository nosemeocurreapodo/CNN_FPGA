import torch
from torchvision import datasets, transforms


class mnist_dataset:
    def __init__(self, batch_size):    
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
        test_dataset  = datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        self.classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'sis', 'seven', 'eight', 'nine')

        self.input_size = (1, 32, 32)

 
class fmnist_dataset:
    def __init__(self, batch_size):   
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        self.classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'sis', 'seven', 'eight', 'nine')

        self.input_size = (1, 32, 32)


class cifar10_dataset:
    def __init__(self, batch_size):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                    (0.2470, 0.2435, 0.2616))  # mean, std
        ])

        # Transformations for testing: just convert and normalize
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                    (0.2470, 0.2435, 0.2616))
        ])

        train_dataset = datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=test_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse',
                        'ship', 'truck')

        self.input_size = (3, 32, 32)


class cifar100_dataset:
    def __init__(self, batch_size):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                    (0.2470, 0.2435, 0.2616))  # mean, std
        ])

        # Transformations for testing: just convert and normalize
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                    (0.2470, 0.2435, 0.2616))
        ])

        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)

        self.classes = [x for x in range(100)]

        self.input_size = (3, 32, 32)


class imagenet_dataset:   
    def __init__(self, batch_size):  
        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                    (0.2470, 0.2435, 0.2616))  # mean, std
        ])

        # Transformations for testing: just convert and normalize
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                    (0.2470, 0.2435, 0.2616))
        ])

        train_dataset = datasets.ImageNet(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.ImageNet(root='./data', train=False, download=True, transform=test_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=2)

        self.classes = [x for x in range(100)]

        self.input_size = (3, 32, 32)
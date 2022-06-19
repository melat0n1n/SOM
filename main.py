import os
import time
import json
import torch
import utils
import argparse
import glob, os
import matplotlib
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from matplotlib.widgets import RadioButtons
import numpy as np


from torch.utils.data.dataloader import DataLoader

from som import SOM

#device = 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Set args
    parser = argparse.ArgumentParser(description='Self Organizing Map')
    parser.add_argument('--color', dest='dataset', action='store_const',
                        const='color', default=None,
                        help='use color')
    parser.add_argument('--mnist', dest='dataset', action='store_const',
                        const='mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--fashion_mnist', dest='dataset', action='store_const',
                        const='fashion_mnist', default=None,
                        help='use mnist dataset')
    parser.add_argument('--yaMaps', dest='dataset', action='store_const',
                        const='yaMaps', default=None,
                        help='use satellite images dataset')
    parser.add_argument('--train', action='store_const',
                        const=True, default=False,
                        help='train network')
    parser.add_argument('--test', action='store_const',
                        const=True, default=False,
                        help='test network')
    parser.add_argument('--prepare_dataset', action='store_const',
                        const=True, default=False,
                        help='split dataset images into tiles 32x32')
    parser.add_argument('--inspect', action='store_const',
                        const=True, default=False,
                        help='inspect SOM neurons')
    parser.add_argument('--mode', type=int, default=1, help='classifying mode')
    parser.add_argument('--data_step', type=int, default=1, help='step for parsing: data = 60k/data_step')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.2, help='input learning rate')
    parser.add_argument('--epoch', type=int, default=1000, help='input total epoch')
    parser.add_argument('--data_dir', type=str, default='datasets', help='set a data directory')
    parser.add_argument('--res_dir', type=str, default='results', help='set a result directory')
    parser.add_argument('--model_dir', type=str, default='model', help='set a model directory')
    parser.add_argument('--row', type=int, default=10, help='set SOM row length')
    parser.add_argument('--col', type=int, default=10, help='set SOM col length')
    args = parser.parse_args()

    # Hyper parameters
    DATA_DIR = args.data_dir + '/' + args.dataset
    RES_DIR = args.res_dir + '/' + args.dataset
    MODEL_DIR = args.model_dir + '/' + args.dataset
    TEST_DATA_DIR = args.data_dir + '/' + args.dataset + '/' + 'test'
    RES_DATA_DIR = args.res_dir + '/' + args.dataset + '/' + 'test'

    # Create results dir
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # Create results/datasetname dir
    if not os.path.exists(RES_DIR):
        os.makedirs(RES_DIR)

    # Create model dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Create model/datasetname dir
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    dataset = args.dataset
    batch_size = args.batch_size
    total_epoch = args.epoch
    row = args.row
    col = args.col
    train = args.train
    test = args.test
    data_step = args.data_step
    mode = args.mode
    prepare_ds = args.prepare_dataset
    inspect = args.inspect

    # Split dataset images into tiles
    if prepare_ds is True:
        for file in glob.glob('*.jpg', root_dir=DATA_DIR):
            utils.tile(file, DATA_DIR, DATA_DIR + '/' + 'root' + '/' + 'set', 32)
        print('Done!')
        exit(0)


    # Create som instance of SOM class
    som = SOM(input_size=32 * 32 * 3, out_size=(row, col))
    # Load saved state
    print('Building Model...')
    if os.path.exists('%s/som.pth' % MODEL_DIR):
        som.load_state_dict(torch.load('%s/som.pth' % MODEL_DIR))
        print('Model Loaded!')
    else:
        print('Create Model!')
    som = som.to(device)

    if (inspect is True) and (dataset == 'yaMaps'):
        # Create classes
        class_num = int(input('Enter the number of classes: '))
        word_dict = {}
        color_dict = {}
        print('Input class and then color separated by space')
        for i in range(1, class_num+1):
            word, color = input(str(i) + ' - ').split()
            word_dict[word] = i
            color_dict[i] = color

        # Save classes in txt for future use in test module
        with open(RES_DIR + '/' + 'convert.txt', 'w') as convert_file:
            convert_file.write(json.dumps(color_dict))


        # Plot current weights of neurons on subplots
        print('Close window after selecting, classes will be saved automatically')
        neurons = som.get_weights((3, 32, 32))
        fig = plt.figure(figsize=(row, col))
        plots = list()
        for i in range(1, len(neurons)+1):
            img = torch.permute(neurons[i-1], (1, 2, 0))
            plots.append(fig.add_subplot(row, col, i))
            plt.imshow(img)
            plt.axis('off')

        # Create radio button menu
        rax = plt.axes([0.00, 0.4, 0.1, 0.15])
        labels = [key for key in word_dict]
        radio = RadioButtons(rax, labels)

        # Redraw radio button menu after click
        def func(label):
            plt.draw()

        # Add title to the neuron after click
        def onclick(event):
            for j, ax in enumerate(plots):
                if ax == event.inaxes:
                    print('Neuron #', j)
                    ax.title.set_text(radio.value_selected)
                    print('Before change:', som.get_type(j).item())
                    som.set_type(j, torch.tensor(word_dict[radio.value_selected]))
                    print('After change:', som.get_type(j).item())
                    plt.draw()
                    torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)


        # Setup onclick events for radio buttons and subplot tiles
        radio.on_clicked(func)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()



    # Dataset creation
    # For mnist and fashion_mnist datasets for train and test are the same
    # For yaMaps there are different datasets
    if (train or test) is True:
        # This module call color.py which is running autonomously
        if dataset == 'color' and train is True:
            import color
            exit(0)

        # Add any transform for input image such as scale or random crop if you want
        transform = transforms.Compose(
                [transforms.ToTensor()]
        )
        list()

        if dataset == 'mnist':
            dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
            sub_set = torch.utils.data.Subset(dataset, range(0, len(dataset), data_step))
            data_loader = DataLoader(sub_set, batch_size=batch_size, shuffle=True)
        elif dataset == 'fashion_mnist':
            dataset = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
            sub_set = torch.utils.data.Subset(dataset, range(0, len(dataset), data_step))
            data_loader = DataLoader(sub_set, batch_size=batch_size, shuffle=True)
        elif dataset == 'yaMaps' and test is False:
            dataset = datasets.ImageFolder(DATA_DIR + '/' + 'root', transform=transform)
            sub_set = torch.utils.data.Subset(dataset, range(0, len(dataset), data_step))
            data_loader = DataLoader(sub_set, batch_size=batch_size, shuffle=True)


    if train is True:
        losses = list()
        for epoch in range(total_epoch):
            running_loss = 0
            start_time = time.time()
            for idx, (X, Y) in enumerate(data_loader):
                X = X.view(-1, 32 * 32 * 3).to(device)    # flatten
                loss = som.self_organizing(X, Y, epoch, total_epoch, mode)    # train som
                running_loss += loss

            losses.append(running_loss)
            print('epoch = %d, loss = %.2f, time = %.2fs' % (epoch + 1, running_loss, time.time() - start_time))

            if epoch % 500 == 0:
                if dataset == 'mnist':
                    som.save_result_digits('%s/digits_%d.png' % (RES_DIR, epoch))
                som.save_result('%s/som_epoch_%d.png' % (RES_DIR, epoch), (3, 32, 32))
                torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)

        torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)
        plt.title('SOM loss')
        plt.plot(losses)
        plt.show()

    # Run test for nmist and fashion_mnist datasets
    if (test is True) and (dataset == 'mnist' or dataset == 'fashion_mnist'):
        avg_error = 0
        # Run test and count average error
        for idx, (X, Y) in enumerate(data_loader):
            X = X.view(-1, 32 * 32 * 3).to(device)
            avg_error = som.test(X, Y, avg_error)
        print('error = %.2f' % avg_error)

    if (test is True) and (dataset == 'yaMaps'):
        # Split input image into tiles
        for file in glob.glob('*.jpg', root_dir=TEST_DATA_DIR):
            utils.tile(file, TEST_DATA_DIR, TEST_DATA_DIR + '/' + 'root' + '/' + 'set', 32)
        # Create dataset from that tiles
        dataset = datasets.ImageFolder(TEST_DATA_DIR + '/' + 'root', transform=transforms.ToTensor())
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Create list of detected classes
        bmu_color = list()
        # Load saved classes
        with open(RES_DIR + '/' + 'convert.txt') as f:
            data = f.read()
        color_dict = json.loads(data)
        # Load that classes into som instance
        som.load_class_color(color_dict)
        # Run test
        for idx, (X, Y) in enumerate(data_loader):
            X = X.view(-1, 32 * 32 * 3).to(device)
            som.test_map(X, bmu_color)

        # Plot image with detected classes (almost hardcode)
        fig, ax = plt.subplots(1, 1, squeeze=True, figsize=[5, 5])
        ax.set_xlim([0, 8]) # for input image 256x256 and tile size 32x32 (8 tiles in row)
        ax.set_ylim([0, 8]) # for input image 256x256 and tile size 32x32 (8 tiles in column)
        x = 0 # counter for element in a row
        y = 7 # counter for rows
        for i in range(0, 64): # 8x8 = 64 tiles
            rect = matplotlib.patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=bmu_color[i],
                                                facecolor=bmu_color[i])
            ax.add_patch(rect)
            # Move coordinate for patching
            x = x + 1
            if (i + 1) % 8 == 0 and i != 0:
                y = y - 1
                x = 0

        plt.show()

    # Save digits for mnist dataset
    if dataset == 'mnist':
        som.save_result_digits('%s/digits_result.png' % (RES_DIR))

    # Save result weights
    som.save_result('%s/som_result.png' % (RES_DIR), (3, 32, 32))
    torch.save(som.state_dict(), '%s/som.pth' % MODEL_DIR)

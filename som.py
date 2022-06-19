import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image


class SOM(nn.Module):
    def __init__(self, input_size, out_size=(10, 10), lr=0.3, sigma=None):
        '''

        :param input_size:
        :param out_size:
        :param lr:
        :param sigma:
        '''
        super(SOM, self).__init__()
        self.input_size = input_size
        self.out_size = out_size

        self.lr = lr
        if sigma is None:
            self.sigma = max(out_size) / 2
        else:
            self.sigma = float(sigma)

        self.class_colors = {}

        self.weight = nn.Parameter(torch.randn(input_size, out_size[0] * out_size[1]), requires_grad=False)
        self.types = nn.Parameter(torch.zeros(10, out_size[0] * out_size[1]), requires_grad=False)
        self.class_param = nn.Parameter(torch.zeros(out_size[0] * out_size[1], 1), requires_grad=False)
        self.locations = nn.Parameter(torch.Tensor(list(self.get_map_index())), requires_grad=False)
        self.pdist_fn = nn.PairwiseDistance(p=2)


    def get_map_index(self):
        '''Two-dimensional mapping function'''
        for x in range(self.out_size[0]):
            for y in range(self.out_size[1]):
                yield (x, y)

    def _neighborhood_fn(self, input, current_sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(current_sigma ** 2)
        input.neg_()
        input.exp_()

        return input

    def forward(self, input):
        '''
        Find the location of best matching unit.
        :param input: data
        :return: location and index of best matching unit, loss
        '''
        batch_size = input.size()[0]

        input = input.view(batch_size, 1, -1) # swap -1 and 1
        batch_weight = self.weight.expand(batch_size, -1, -1)
        batch_weight = batch_weight.transpose(1, 2) # add transpose

        dists = self.pdist_fn(input, batch_weight)
        dists = dists.view(batch_size, -1) # new feature!
        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1, keepdim=True)

        bmu_locations = self.locations[bmu_indexes]
        return bmu_indexes, bmu_locations, losses.sum().div_(batch_size).item()

    def load_class_color(self, colors):
        '''
        Set colors for classes
        :param colors: dictionary of class colors which were saved in txt
        :return:
        '''
        for key, value in colors.items():
            self.class_colors[key] = value

    def set_type(self, index, value):
        '''
        Set type of class (number)
        :param index: index of neuron
        :param value: class (number)
        :return:
        '''
        self.class_param[index] = value

    def get_type(self, index):
        '''
        Get class value of neuron
        :param index: index of neuron
        :return: class (number)
        '''
        return self.class_param[index]

    def classify(self, digits, delta, mode):
        '''
        Classify SOM neurons
        :param digits: set of digits in dataset
        :param delta: delta value for neurons
        :return:
        '''
        if mode == 1:
            for pic in range(digits.size(dim=0)):
                self.types[int(digits[pic].item())].add_(delta[pic])


    def self_organizing(self, input, digits, current_iter, max_iter, mode):
        '''
        Train the Self Oranizing Map(SOM)
        :param input: training data
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''
        batch_size = input.size()[0]
        #Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = self.lr * iter_correction
        sigma = self.sigma * iter_correction
        #Find best matching unit
        bmu_indexes, bmu_locations, loss = self.forward(input)

        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)

        lr_locations = self._neighborhood_fn(distance_squares, sigma)

        lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr_locations * (input.unsqueeze(2) - self.weight)
        lr_locations.squeeze_(1)

        #if current_iter >= (max_iter/2):
        self.classify(digits, lr_locations, mode)

        delta = delta.sum(dim=0)
        delta.div_(batch_size)
        self.weight.data.add_(delta)

        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])
        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])

    def get_weights(self, im_size=(0, 0, 0)):
        '''
        Get tensor of weights which are represented as image
        :param im_size: image size
        :return: tensor of images
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])
        images = images.permute(3, 0, 1, 2)
        return images

    def save_result_digits(self, dir):
        '''
        Plot SOM's neuron values
        :param dir: directory to save
        :return:
        '''
        val, result_digits = self.types.max(dim=0)
        plt.figure(1, figsize=(15, 15))
        plt.scatter(list(range(self.out_size[0])), list(range(self.out_size[1])))
        k = 0
        for j in range(self.out_size[0]-1, -1, -1):
            for i in range(self.out_size[1]):
                plt.text(i, j, int(result_digits[k].item()), ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.5, lw=0), size='xx-large')
                k += 1

        plt.savefig(dir, dpi='figure', format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None, )

    def test(self, input, digits, avg_error):
        '''
        Test function
        :param input: batch of images
        :param digits: batch of values
        :param avg_error: average error
        :return: updated average error
        '''
        batch_size = input.size()[0]
        bmu_indexes, bmu_locations, loss = self.forward(input)
        val, result_digits = self.types.max(dim=0)
        false_counter = 0
        for i in range(batch_size):
            if (digits[i].item() != result_digits[bmu_indexes[i].item()].item()) is True:
                false_counter += 1

        error = false_counter/batch_size
        avg_error = (avg_error + error)/2
        return avg_error

    def test_map(self, input, bmu_color):
        '''
        Test map picture
        :param input: batch of images
        :param bmu_color: list of registered colors
        :return: updated list of registered colors
        '''
        batch_size = input.size()[0]
        bmu_indexes, bmu_locations, loss = self.forward(input)

        for i in range(batch_size):
            bmu_class = str(int(self.class_param[bmu_indexes[i].item()].item()))
            value = self.class_colors[str(bmu_class)]
            bmu_color.append(value)
        return bmu_color

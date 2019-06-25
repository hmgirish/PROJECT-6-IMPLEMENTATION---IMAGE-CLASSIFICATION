Image Classification

In this project, you'll classify images from the CIFAR-10 dataset
(https://www.cs.toronto.edu/~kriz/cifar.html). The dataset consists of airplanes, dogs, cats, and other
objects. You'll preprocess the images, then train a convolutional neural network on all the samples.
The images need to be normalized and the labels need to be one-hot encoded. You'll get to apply
what you learned and build a convolutional, max pooling, dropout, and fully connected layers. At the
end, you'll get to see your neural network's predictions on the sample images.

Get the Data

Run the following cell to download the CIFAR-10 dataset for python
(https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

Data

CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the
80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10
object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and
Geoffrey Hinton.

Let's get the data by running the following function

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0
    
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
        
if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset
    urlretrieve(
        'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        'cifar-10-python.tar.gz',
        pbar.hook)
        
if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

Explore the Data

The dataset is broken into batches to prevent your machine from running out of memory. The
CIFAR-10 dataset consists of 5 batches, named data_batch_1, data_batch_2, etc.. Each batch
contains the labels and images that are one of the following:

airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck

Understanding a dataset is part of making predictions on the data. Play around with the code cell
below by changing the batch_id and sample_id. The batch_id is the id for a batch (1-5). The
sample_id is the id for a image and label pair in the batch.

Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are
the labels in order or random?". Answers to questions like these will help you preprocess the data
and end up with better predictions.

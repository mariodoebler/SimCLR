from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from itertools import chain
import torch

class AtariDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, episodes, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        self.episodes = episodes
        self.flat_episodes = list(chain.from_iterable(self.episodes))
        self.transform = transform

    def __len__(self):
        return len(self.flat_episodes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        framestack = self.flat_episodes[idx]
        # framestack_im = transforms.ToPILImage()(framestack) #.convert("RGB")
        # print(im)
        # print(framestack_im.size)
        # print(framestack.shape)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        samples = self.transform(framestack)

        return samples

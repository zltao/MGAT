import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DataLoad(Dataset):
	def __init__(self, path, num_user, num_item):
		super(DataLoad, self).__init__()
		self.data = np.load(path+'train.npy')
		self.adj_lists = np.load(path+'final_adj_dict.npy').item()
		self.all_set = set(range(num_user, num_user+num_item))

	def __getitem__(self, index):
		user, pos_item = self.data[index]
		neg_item = random.sample(self.all_set.difference(self.adj_lists[user]), 1)[0]
		return [user, pos_item, neg_item]

	def __len__(self):
		return len(self.data)



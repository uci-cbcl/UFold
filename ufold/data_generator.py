import _pickle as cPickle
import collections
import pdb
from collections import Counter
from itertools import product
from multiprocessing import Pool
from random import shuffle

from torch.utils import data

from ufold.utils import *

perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1, 3], [3, 1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]


class RNASSDataGenerator(object):
    def __init__(self, data_dir, split, upsampling=False):
        self.data_dir = data_dir
        self.split = split
        self.upsampling = upsampling
        # Load vocab explicitly when needed
        self.load_data()
        # Reset batch pointer to zero
        self.batch_pointer = 0

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple("RNA_SS_data", "seq ss_label length name pairs")
        with open(os.path.join(data_dir, "%s" % self.split), "rb") as f:
            self.data = cPickle.load(f, encoding="iso-8859-1")
        if self.upsampling:
            self.data = self.upsampling_data_new()
        self.data_x = np.array([instance[0] for instance in self.data])
        self.data_y = np.array([instance[1] for instance in self.data])
        self.pairs = np.array([instance[-1] for instance in self.data])
        # pdb.set_trace()
        self.seq_length = np.array([instance[2] for instance in self.data])
        self.len = len(self.data)
        self.seq = list(p.map(encoding2seq, self.data_x))
        self.seq_max_len = len(self.data_x[0])
        self.data_name = np.array([instance[3] for instance in self.data])
        # self.matrix_rep = np.array(list(p.map(creatmat, self.seq)))
        # self.matrix_rep = np.zeros([self.len, len(self.data_x[0]), len(self.data_x[0])])

    def upsampling_data(self):
        pdb.set_trace()
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split("/")[2], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]
            data_list.append(data[index])
        final_d_list = list()
        # for d in data_list:
        #     index = np.random.choice(d.shape[0], max_num)
        #     final_d_list += list(d[index])
        for i in [0, 1, 5, 7]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num)
            final_d_list += list(d[index])

        for i in [2, 3, 4]:
            d = data_list[i]
            index = np.random.choice(d.shape[0], max_num * 2)
            final_d_list += list(d[index])

        d = data_list[6]
        index = np.random.choice(d.shape[0], int(max_num / 2))
        final_d_list += list(d[index])

        shuffle(final_d_list)
        return final_d_list

    def upsampling_data_new(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split("_")[0], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]
            data_list.append(data[index])
        final_d_list = list()
        for d in data_list:
            final_d_list += list(d)
            if d.shape[0] < 300:
                index = np.random.choice(d.shape[0], 300 - d.shape[0])
                final_d_list += list(d[index])
            if d.shape[0] == 652:
                print("processing PDB seq...")
                index = np.random.choice(d.shape[0], d.shape[0] * 4)
                final_d_list += list(d[index])
        shuffle(final_d_list)
        return final_d_list

    def upsampling_data_new_addPDB(self):
        name = [instance.name for instance in self.data]
        d_type = np.array(list(map(lambda x: x.split("_")[0], name)))
        data = np.array(self.data)
        max_num = max(Counter(list(d_type)).values())
        data_list = list()
        for t in sorted(list(np.unique(d_type))):
            index = np.where(d_type == t)[0]
            data_list.append(data[index])
        final_d_list = list()
        for d in data_list:
            final_d_list += list(d)
            if d.shape[0] < 300:
                index = np.random.choice(d.shape[0], 300 - d.shape[0])
                final_d_list += list(d[index])
            if d.shape[0] == 652:
                print("processing PDB seq...")
                index = np.random.choice(d.shape[0], d.shape[0] * 4)
                final_d_list += list(d[index])
        shuffle(final_d_list)
        return final_d_list

    def next_batch(self, batch_size):
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        batch_x = self.data_x[bp : bp + batch_size]
        batch_y = self.data_y[bp : bp + batch_size]
        batch_seq_len = self.seq_length[bp : bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0

        yield batch_x, batch_y, batch_seq_len

    def pairs2map(self, pairs):
        seq_len = self.seq_max_len
        contact = np.zeros([seq_len, seq_len])
        for pair in pairs:
            contact[pair[0], pair[1]] = 1
        return contact

    def next_batch_SL(self, batch_size):
        p = Pool()
        bp = self.batch_pointer
        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        data_y = self.data_y[bp : bp + batch_size]
        data_seq = self.data_x[bp : bp + batch_size]
        data_pairs = self.pairs[bp : bp + batch_size]

        self.batch_pointer += batch_size
        if self.batch_pointer >= len(self.data_x):
            self.batch_pointer = 0
        contact = np.array(list(map(self.pairs2map, data_pairs)))
        matrix_rep = np.zeros(contact.shape)
        yield contact, data_seq, matrix_rep

    def get_one_sample(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        # data_len = np.nonzero(self.data_x[index].sum(axis=2))[0].max()
        data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact = self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name

    def get_one_sample_long(self, index):
        data_y = self.data_y[index]
        data_seq = self.data_x[index]
        # pdb.set_trace()
        # print(data_seq.shape)
        data_len = np.nonzero(self.data_x[index].sum(axis=1))[0].max()
        # data_len = self.seq_length[index]
        data_pair = self.pairs[index]
        data_name = self.data_name[index]

        contact = self.pairs2map(data_pair)
        matrix_rep = np.zeros(contact.shape)
        return contact, data_seq, matrix_rep, data_len, data_name

    def random_sample(self, size=1):
        # random sample one RNA
        # return RNA sequence and the ground truth contact map
        index = np.random.randint(self.len, size=size)
        data = list(np.array(self.data)[index])
        data_seq = [instance[0] for instance in data]
        data_stru_prob = [instance[1] for instance in data]
        data_pair = [instance[-1] for instance in data]
        seq = list(map(encoding2seq, data_seq))
        contact = list(map(self.pairs2map, data_pair))
        return contact, seq, data_seq

    def get_one_sample_cdp(self, index):
        data_seq = self.data_x[index]
        data_label = self.data_y[index]

        return data_seq, data_label


class RNASSDataGenerator_input(object):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.load_data()

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        RNA_SS_data = collections.namedtuple("RNA_SS_data", "seq ss_label length name pairs")
        input_file = open(os.path.join(data_dir, "%s.txt" % self.split), "r").readlines()
        self.data_name = np.array([itm.strip()[1:] for itm in input_file if itm.startswith(">")])
        self.seq = [
            itm.strip().upper().replace("T", "U")
            for itm in input_file
            if itm.upper().startswith(("A", "U", "C", "G", "T"))
        ]
        self.len = len(self.seq)
        self.seq_length = np.array([len(item) for item in self.seq])
        self.data_x = np.array([self.one_hot_600(item) for item in self.seq])
        self.seq_max_len = 600
        self.data_y = self.data_x

    def one_hot_600(self, seq_item):
        RNN_seq = seq_item
        BASES = "AUCG"
        bases = np.array([base for base in BASES])
        feat = np.concatenate(
            [
                [(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)])
                for base in RNN_seq
            ]
        )
        if len(seq_item) <= 600:
            one_hot_matrix_600 = np.zeros((600, 4))
        else:
            one_hot_matrix_600 = np.zeros((600, 4))
            # one_hot_matrix_600 = np.zeros((len(seq_item),4))
        one_hot_matrix_600[
            : len(seq_item),
        ] = feat
        return one_hot_matrix_600

    def get_one_sample(self, index):

        # This will return a smaller size if not sufficient
        # The user must pad the batch in an external API
        # Or write a TF module with variable batch size
        # data_y = self.data_y[index]
        data_seq = self.data_x[index]
        data_len = self.seq_length[index]
        # data_pair = self.pairs[index]
        data_name = self.data_name[index]

        # contact= self.pairs2map(data_pair)
        # matrix_rep = np.zeros(contact.shape)
        # return contact, data_seq, matrix_rep, data_len, data_name
        return data_seq, data_len, data_name


# using torch data loader to parallel and speed up the data load process
class Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def merge_data(self):
        return self.data2

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        return self.data.get_one_sample(index)


class Dataset_1800(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contacts, seq_embeddings, matrix_reps, seq_lens = self.data.get_one_sample(index)
        PE = get_pe(torch.Tensor([seq_lens]).long(), 1800).numpy()
        PE = torch.Tensor(PE[0]).float()
        small_seqs, comb_index_1 = get_chunk_combination(torch.Tensor(seq_embeddings).float())
        PE_small_seqs, comb_index_2 = get_chunk_combination(PE)
        contacts_b = get_chunk_gt(torch.Tensor(contacts).float(), comb_index_1)

        assert comb_index_1 == comb_index_2

        seq_embedding_batch = torch.cat([seq.unsqueeze_(0) for seq in small_seqs], 0).float()
        PE_batch = torch.cat([pe.unsqueeze_(0) for pe in PE_small_seqs], 0).float()
        contacts_batch = torch.cat([contact.unsqueeze_(0) for contact in contacts_b], 0).float()

        return seq_embedding_batch, PE_batch, contacts_batch, comb_index_1, seq_embeddings, contacts, seq_lens


class Dataset_cdp(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        return self.data.get_one_sample_cdp(index)


class Dataset_FCN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        pdb.set_trace()
        print("get_data...")
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = data_seq.shape[0]
        data_fcn = np.zeros((16, l, l))
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n] = np.matmul(data_seq[:, i].reshape(-1, 1), data_seq[:, j].reshape(1, -1))
        return contact, data_fcn, matrix_rep, data_len, data_seq, data_name


class Dataset_FCN_input(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        pdb.set_trace()
        print("get_data...")
        data_seq, data_len, data_name = self.data.get_one_sample(index)
        l = data_seq.shape[0]
        data_fcn = np.zeros((16, l, l))
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n] = np.matmul(data_seq[:, i].reshape(-1, 1), data_seq[:, j].reshape(1, -1))
        return data_fcn, data_len, data_seq, data_name


class Dataset_Cut(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        if l >= 600:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        return contact[:l, :l], data_fcn, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_8(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((10, l, l))
        if l >= 600:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        n_dim = 0
        for i in range(4):
            for j in range(i, 4):
                if i != j:
                    data_fcn[n_dim, :data_len, :data_len] = np.matmul(
                        data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
                    ) + np.matmul(data_seq[:data_len, j].reshape(-1, 1), data_seq[:data_len, i].reshape(1, -1))
                else:
                    data_fcn[n_dim, :data_len, :data_len] = np.matmul(
                        data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
                    )
                n_dim += 1
        """
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
        """
        return contact[:l, :l], data_fcn, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_outer(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 600:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:, :data_len, :data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape(
            (-1, data_len, data_len)
        )
        # feature = np.concatenate((data_fcn,feature),axis=0)
        return contact[:l, :l], feature, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_concat(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:, :data_len, :data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape(
            (-1, data_len, data_len)
        )
        feature = np.concatenate((data_fcn, feature), axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_concat_new(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        data_seq, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            # contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            # contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(
            data_seq[
                :data_len,
            ]
        )
        # zero_mask = z_mask(data_len)[None, :, :, None]
        # label_mask = l_mask(data_seq, data_len)
        # temp = data_seq[None, :data_len, :data_len]
        # temp = np.tile(temp, (temp.shape[1], 1, 1))
        # feature[:,:data_len,:data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape((-1,data_len,data_len))
        # feature = np.concatenate((data_fcn,feature),axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return data_fcn_2, data_len, data_seq[:l], data_name


class Dataset_Cut_concat_new_merge(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data1, data2, data3):
        "Initialization"
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.merge_data()
        self.data = self.data2

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def merge_data(self):
        self.data2.data_x = np.concatenate((self.data1.data_x, self.data2.data_x, self.data3.data_x), axis=0)
        self.data2.data_y = np.concatenate((self.data1.data_y, self.data2.data_y, self.data3.data_y), axis=0)
        self.data2.seq_length = np.concatenate(
            (self.data1.seq_length, self.data2.seq_length, self.data3.seq_length), axis=0
        )
        self.data2.pairs = np.concatenate((self.data1.pairs, self.data2.pairs, self.data3.pairs), axis=0)
        self.data2.data_name = np.concatenate(
            (self.data1.data_name, self.data2.data_name, self.data3.data_name), axis=0
        )
        self.data2.len = len(self.data2.data_name)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(
            data_seq[
                :data_len,
            ]
        )
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:, :data_len, :data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape(
            (-1, data_len, data_len)
        )
        feature = np.concatenate((data_fcn, feature), axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_concat_new_merge_multi(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_list):
        "Initialization"
        self.data2 = data_list[0]
        if len(data_list) > 1:
            self.data = self.merge_data(data_list)
        else:
            self.data = self.data2

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def merge_data(self, data_list):

        self.data2.data_x = np.concatenate((data_list[0].data_x, data_list[1].data_x), axis=0)
        self.data2.data_y = np.concatenate((data_list[0].data_y, data_list[1].data_y), axis=0)
        self.data2.seq_length = np.concatenate((data_list[0].seq_length, data_list[1].seq_length), axis=0)
        self.data2.pairs = np.concatenate((data_list[0].pairs, data_list[1].pairs), axis=0)
        self.data2.data_name = np.concatenate((data_list[0].data_name, data_list[1].data_name), axis=0)
        for item in data_list[2:]:
            self.data2.data_x = np.concatenate((self.data2.data_x, item.data_x), axis=0)
            self.data2.data_y = np.concatenate((self.data2.data_y, item.data_y), axis=0)
            self.data2.seq_length = np.concatenate((self.data2.seq_length, item.seq_length), axis=0)
            self.data2.pairs = np.concatenate((self.data2.pairs, item.pairs), axis=0)
            self.data2.data_name = np.concatenate((self.data2.data_name, item.data_name), axis=0)

        self.data2.len = len(self.data2.data_name)
        return self.data2

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(
            data_seq[
                :data_len,
            ]
        )
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:, :data_len, :data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape(
            (-1, data_len, data_len)
        )
        feature = np.concatenate((data_fcn, feature), axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn_2, data_fcn_1, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_concat_new_merge_two(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data1, data2):
        "Initialization"
        self.data1 = data1
        self.data2 = data2
        # self.data3 = data3
        self.merge_data()
        self.data = self.data2

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def merge_data(self):
        self.data2.data_x = np.concatenate((self.data1.data_x[:, :600, :], self.data2.data_x), axis=0)
        self.data2.data_y = np.concatenate((self.data1.data_y[:, :600, :], self.data2.data_y), axis=0)
        self.data2.seq_length = np.concatenate((self.data1.seq_length, self.data2.seq_length), axis=0)
        self.data2.pairs = np.concatenate((self.data1.pairs, self.data2.pairs), axis=0)
        self.data2.data_name = np.concatenate((self.data1.data_name, self.data2.data_name), axis=0)
        self.data2.len = len(self.data2.data_name)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(
            data_seq[
                :data_len,
            ]
        )
        zero_mask = z_mask(data_len)[None, :, :, None]
        label_mask = l_mask(data_seq, data_len)
        temp = data_seq[None, :data_len, :data_len]
        temp = np.tile(temp, (temp.shape[1], 1, 1))
        feature[:, :data_len, :data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape(
            (-1, data_len, data_len)
        )
        feature = np.concatenate((data_fcn, feature), axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_concat_new_canonicle(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        # contact, data_seq, matrix_rep, data_len, data_name, data_pair = self.data.get_one_sample_addpairs(index)
        l = get_cut_len(data_len, 80)
        data_fcn = np.zeros((16, l, l))
        # data_nc = np.zeros((2, l, l))
        data_nc = np.zeros((10, l, l))
        feature = np.zeros((8, l, l))
        if l >= 500:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        for n, cord in enumerate(perm_nc):
            i, j = cord
            data_nc[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        data_nc = data_nc.sum(axis=0).astype(np.bool)
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(
            data_seq[
                :data_len,
            ]
        )
        # zero_mask = z_mask(data_len)[None, :, :, None]
        # label_mask = l_mask(data_seq, data_len)
        # temp = data_seq[None, :data_len, :data_len]
        # temp = np.tile(temp, (temp.shape[1], 1, 1))
        # feature[:,:data_len,:data_len] = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2).reshape((-1,data_len,data_len))
        # feature = np.concatenate((data_fcn,feature),axis=0)
        # return contact[:l, :l], data_fcn, feature, matrix_rep, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, data_fcn, matrix_rep, data_len, data_seq[:l], data_name
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        # return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc, data_pair
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name, data_nc, l


class Dataset_Cut_input(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        data_seq, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 160)
        data_fcn = np.zeros((16, l, l))
        if l >= 600:
            contact_adj = np.zeros((l, l))
            # contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            # contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        return data_fcn, data_len, data_seq[:l], data_name
        # return contact[:l, :l], data_fcn, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_long(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample(index)
        l = get_cut_len(data_len, 160)
        data_fcn = np.zeros((16, l, l))
        if l >= 1800:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        return contact[:l, :l], data_fcn, matrix_rep, data_len, data_seq[:l], data_name


class Dataset_Cut_long_17dim(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data):
        "Initialization"
        self.data = data

    def __len__(self):
        "Denotes the total number of samples"
        return self.data.len

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        contact, data_seq, matrix_rep, data_len, data_name = self.data.get_one_sample_long(index)
        if data_len > 1500:
            data_len = 1500
        l = get_cut_len(data_len, 160)
        data_fcn = np.zeros((16, l, l))
        if l >= 1800:
            contact_adj = np.zeros((l, l))
            contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
            contact = contact_adj
            seq_adj = np.zeros((l, 4))
            seq_adj[:data_len] = data_seq[:data_len]
            data_seq = seq_adj
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n, :data_len, :data_len] = np.matmul(
                data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1)
            )
        data_fcn_1 = np.zeros((1, l, l))
        data_fcn_1[0, :data_len, :data_len] = creatmat(
            data_seq[
                :data_len,
            ]
        )
        data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
        return contact[:l, :l], data_fcn_2, matrix_rep, data_len, data_seq[:l], data_name


def get_cut_len(data_len, set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


def z_mask(seq_len):
    mask = np.ones((seq_len, seq_len))
    return np.triu(mask, 2)


def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)


def paired(x, y):
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0


def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat

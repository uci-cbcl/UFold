import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

import pdb
import subprocess

# import sys
# sys.path.append('./..')


# from FCN import FCNNet
from Network import U_Net as FCNNet
#from Network3 import U_Net_FP as FCNNet

from ufold.utils import *
from ufold.config import process_config
from ufold.postprocess import postprocess_new as postprocess

from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_merge as Dataset_FCN_merge
from ufold.data_generator import Dataset_Cut_concat_new_merge_multi as Dataset_FCN_merge
import collections


def train(contact_net,train_merge_generator,epoches_first):
    epoch = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    u_optimizer = optim.Adam(contact_net.parameters())
    
    # test_script()
    #t1 = subprocess.getstatusoutput('awk \'{if($1 ~ /^>/)print}\' /data2/darren/experiment/ufold/data/rnastralign_all/rnastralign_train_no_redundant.seq.cdhit')
    #all_cdhit_names = t1[1].split('\n')
    pdb.set_trace()
    steps_done = 0
    print('start training...')
    # There are three steps of training
    # step one: train the u net
    epoch_rec = []
    for epoch in range(epoches_first):
        contact_net.train()
        # num_batches = int(np.ceil(train_data.len / BATCH_SIZE))
        # for i in range(num_batches):
        #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_generator:
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
        #for contacts, seq_embeddings, seq_embeddings_1, matrix_reps, seq_lens, seq_ori, seq_name in train_generator:
            # contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(train_generator))
    
            '''
            compare_name = '>' + seq_name[0]
            if compare_name not in all_cdhit_names:
                continue
            if seq_lens.cpu()[0] > 1500:
                continue
            '''
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            #seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
            # matrix_reps_batch = torch.unsqueeze(
            #     torch.Tensor(matrix_reps.float()).to(device), -1)
    
            # padding the states for supervised training with all 0s
            # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
            #     seq_len, seq_len]).to(device)
    
    
            # PE_batch = get_pe(seq_lens, seq_len).float().to(device)
            # contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
            
            pred_contacts = contact_net(seq_embedding_batch)
            #pred_contacts = contact_net(seq_embedding_batch,seq_embedding_batch_1)
    
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
    
            # Compute loss
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
    
    
            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done=steps_done+1
    
        print('Training log: epoch: {}, step: {}, loss: {}'.format(
                    epoch, steps_done-1, loss_u))
        #pdb.set_trace()
            # model_eval_all_test()
            # torch.save(contact_net.state_dict(), model_path)
            #torch.save(contact_net.state_dict(), model_path + f'unet_bpTR0_addsimmutate_addmoresimilar_finetune{epoch}.pt')
        if epoch > -1:
                #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_TR0_continuefrom99_{epoch}.pt')
                #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_RNAlign_restart_{epoch}.pt')
                #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_merge_alldata_{epoch}.pt')
                #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_TR0bpnewOriuseMXUnet_{epoch}.pt')
            torch.save(contact_net.state_dict(),  f'../models_ckpt/final_model/for_servermodel/tmp/ufold_train_onalldata_{epoch}.pt')

def main():
    torch.cuda.set_device(1)
    
    args = get_args()
    
    config_file = args.config
    
    config = process_config(config_file)
    print("#####Stage 1#####")
    print('Here is the configuration of this run: ')
    print(config)
    
    pdb.set_trace()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
    
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    #model_path = './models_ckpt/'.format(model_type, data_type,d)
    #model_path = './models_ckpt/final_model/unet_train_on_RNAlign_99.pt'
    epoches_first = config.epoches_first

    train_files = args.train_files
    
    
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_torch()
    
    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
    
    pdb.set_trace()
    train_data_list = []
    for file_item in train_files:
        print('Loading dataset: ',file_item)
        if file_item == 'RNAStralign' or file_item == 'ArchiveII':
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
    print('Data Loading Done!!!')
    #train_data = RNASSDataGenerator('data/{}/'.format(data_type), 'train.pickle', False)
    pdb.set_trace()
    '''
    train_data_pdb = RNASSDataGenerator('data/','pdb_from_yx_672.cPickle',False)
    #train_data_TR0 = RNASSDataGenerator('data/','bpRNA_TR0_ori.cPickle',False)
    train_data_TR0_addsim = RNASSDataGenerator('data/','bpRNA_TR0_addbpnewori_calculatefromcontrafold.cPickle',False)
    #train_data = RNASSDataGenerator('data/','bpRNA12_allfamily_generate_onlyPDB.cPickle')
    test_data_archive = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'test_600.pickle')
    test_data_TS0 = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TS0_ori.cPickle')
    test_data_bpnew = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_new20201015.cPickle')
    test_data_TS1 = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TS1_test_new.cPickle')
    test_data_TS2 = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TS3_test.cPickle')
    test_data_TS3 = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TS2_test.cPickle')
    #train_data_MXUnet = RNASSDataGenerator('data/','bpRNA_TR0_addmutate_calculatefromMXUnet.cPickle')
    train_data_MXUnet = RNASSDataGenerator('data/','bpRNA_TR0_addbpnewori_calculatefromMXUnet.cPickle')
    #train_data = RNASSDataGenerator('data/','bpRNA12_38family_generate_train.cPickle',True)
    '''
    
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    # train_set = Dataset(train_data)
    #train_set = Dataset_FCN(train_data)
    #train_set = Dataset_FCN(train_data_MXUnet)
    #train_generator = data.DataLoader(train_set, **params)
    # val_set = Dataset(val_data)
    #train_merge = Dataset_FCN_merge(train_data,train_data_sim,train_data_TR0)
    #train_merge = Dataset_FCN_merge(train_data,train_data_sim,train_data_TR0)
    pdb.set_trace()
    #train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb])
    #train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb])
    #train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb,test_data_archive,test_data_TS0,test_data_bpnew,test_data_TS1,test_data_TS2,test_data_TS3])
    #train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb,test_data_archive,test_data_TS0,test_data_bpnew,test_data_TS1,test_data_TS2,test_data_TS3])
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
    pdb.set_trace()
    
    contact_net = FCNNet(img_ch=17)
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    
    # contact_net.conv1d2.register_forward_hook(get_activation('conv1d2'))
    
    #if LOAD_MODEL and os.path.isfile(model_path):
    #    print('Loading u net model...')
    #    contact_net.load_state_dict(torch.load(model_path))
    
    
    
    # for 5s
    # pos_weight = torch.Tensor([100]).to(device)
    # for length as 600

    train(contact_net,train_merge_generator,epoches_first)

        

#model_eval_all_test()
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()
#torch.save(contact_net.module.state_dict(), model_path + 'unet_final.pt')
# sys.exit()








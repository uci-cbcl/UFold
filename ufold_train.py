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

torch.cuda.set_device(3)

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
evaluate_epi = 1


steps_done = 0
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_torch()

# for loading data
# loading the rna ss data, the data has been preprocessed
# 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_merge as Dataset_FCN_merge
from ufold.data_generator import Dataset_Cut_concat_new_merge_multi as Dataset_FCN_merge
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')

train_data = RNASSDataGenerator('data/{}/'.format(data_type), 'train.pickle', False)
#train_data = RNASSDataGenerator('data/','bpRNA12_allfamily_generate_addPDB.cPickle',True)
#train_data = RNASSDataGenerator('data/','bpRNA_TR0_andsim_mutate.cPickle',False)
#train_data_sim = RNASSDataGenerator('data/','bpRNA_TR0_andsim_mutate_moredata_oribpnew_short.cPickle',False)
#train_data_sim = RNASSDataGenerator('data/','bpRNA_bpnew_contrafold_pred_data.cPickle',False)
train_data_pdb = RNASSDataGenerator('data/','pdb_from_yx_672.cPickle',False)
#train_data = RNASSDataGenerator('data/','bpRNA_TR0_extract9814.cPickle',False)
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
val_data = RNASSDataGenerator('data/{}/'.format(data_type), 'val.pickle')
# test_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_no_redundant')
test_data = RNASSDataGenerator('data/rnastralign_all/', 'test_no_redundant_600.pickle')

seq_len = train_data.data_y.shape[-2]
print('Max seq length ', seq_len)

# using the pytorch interface to parallel the data generation and model training
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
# train_set = Dataset(train_data)
train_set = Dataset_FCN(train_data)
#train_set = Dataset_FCN(train_data_MXUnet)
train_generator = data.DataLoader(train_set, **params)
# val_set = Dataset(val_data)
val_set = Dataset_FCN(val_data)
val_generator = data.DataLoader(val_set, **params)

# test_set = Dataset(test_data)
test_set = Dataset_FCN(test_data)
test_generator = data.DataLoader(test_set, **params)
#pdb.set_trace()
#train_merge = Dataset_FCN_merge(train_data,train_data_sim,train_data_TR0)
#train_merge = Dataset_FCN_merge(train_data,train_data_sim,train_data_TR0)
pdb.set_trace()
#train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb])
#train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb])
train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb,test_data_archive,test_data_TS0,test_data_bpnew,test_data_TS1,test_data_TS2,test_data_TS3])
train_merge = Dataset_FCN_merge([train_data,train_data_TR0_addsim,train_data_pdb,test_data_archive,test_data_TS0,test_data_bpnew,test_data_TS1,test_data_TS2,test_data_TS3])
train_merge_generator = data.DataLoader(train_merge, **params)
pdb.set_trace()

# seq_len =500

# store the intermidiate activation

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# if model_type =='test_lc':
#     contact_net = ContactNetwork_test(d=d, L=seq_len).to(device)
# if model_type == 'att6':
#     contact_net = ContactAttention(d=d, L=seq_len).to(device)
# if model_type == 'att_simple':
#     contact_net = ContactAttention_simple(d=d, L=seq_len).to(device)    
# if model_type == 'att_simple_fix':
#     contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len, 
#         device=device).to(device)
# if model_type == 'fc':
#     contact_net = ContactNetwork_fc(d=d, L=seq_len).to(device)
# if model_type == 'conv2d_fc':
#     contact_net = ContactNetwork(d=d, L=seq_len).to(device)

contact_net = FCNNet(img_ch=17)
# contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
contact_net.to(device)

# contact_net.conv1d2.register_forward_hook(get_activation('conv1d2'))

#if LOAD_MODEL and os.path.isfile(model_path):
#    print('Loading u net model...')
#    contact_net.load_state_dict(torch.load(model_path))


u_optimizer = optim.Adam(contact_net.parameters())

# for 5s
# pos_weight = torch.Tensor([100]).to(device)
# for length as 600
pos_weight = torch.Tensor([300]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight = pos_weight)


# randomly select one sample from the test set and perform the evaluation

def model_eval_all_test():
    contact_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori in test_generator:
        if batch_n%100==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        # matrix_reps_batch = torch.unsqueeze(
        #     torch.Tensor(matrix_reps.float()).to(device), -1)

        # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
        #     seq_len, seq_len]).to(device)

        # PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 50, 1.0, True)
        map_no_train = (u_no_train > 0.5).float()
        result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train_shift += result_no_train_tmp_shift
        seq_lens_list += list(seq_lens)

    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    nt_shift_p,nt_shift_r,nt_shift_f1 = zip(*result_no_train_shift)  
    
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))

    print('Average testing F1 score with pure post-processing allow shift: ', np.average(nt_shift_f1))

    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))

    print('Average testing precision with pure post-processing allow shift: ', np.average(nt_shift_p))

    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))

    print('Average testing recall with pure post-processing allow shift: ', np.average(nt_shift_r))

    result_dict = dict()
    result_dict['exact_p'] = nt_exact_p
    result_dict['exact_r'] = nt_exact_r
    result_dict['exact_f1'] = nt_exact_f1
    result_dict['shift_p'] = nt_shift_p
    result_dict['shift_r'] = nt_shift_r
    result_dict['shift_f1'] = nt_shift_f1
    result_dict['seq_lens'] = seq_lens_list
    result_dict['exact_weighted_f1'] = np.sum(np.array(nt_exact_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))
    result_dict['shift_weighted_f1'] = np.sum(np.array(nt_shift_f1)*np.array(seq_lens_list)/np.sum(seq_lens_list))

    # with open('../results/rnastralign_short_pure_pp_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)


    # with open('../results/rnastralign_short_greedy_sort_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)
    # with open('../results/fcn_rnastralign_short_greedy_sort_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)


epoch = 0

def test_script():
    print('=============test===============')
    # model_eval()
    model_eval_all_test()
    # torch.save(contact_net.state_dict(), model_path + f'unet{epoch}.pt')
    # torch.save(contact_net.module.state_dict(), model_path + f'unet{epoch}.pt')
    print('=============end test===============')
    sys.exit()

# test_script()
#t1 = subprocess.getstatusoutput('awk \'{if($1 ~ /^>/)print}\' /data2/darren/experiment/ufold/data/rnastralign_all/rnastralign_train_no_redundant.seq.cdhit')
#all_cdhit_names = t1[1].split('\n')
pdb.set_trace()
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

        # print(steps_done)
        # if steps_done % OUT_STEP ==0:
        #     print('Stage 1, epoch: {},step: {}, loss: {}'.format(
        #         epoch, steps_done, loss_u))

        # Optimize the model
        u_optimizer.zero_grad()
        loss_u.backward()
        u_optimizer.step()
        steps_done=steps_done+1

    print('Stage 1, epoch: {}, step: {}, loss: {}'.format(
                epoch, steps_done-1, loss_u))
    #pdb.set_trace()
    if epoch%evaluate_epi==0:
        # model_eval_all_test()
        # torch.save(contact_net.state_dict(), model_path)
        #torch.save(contact_net.state_dict(), model_path + f'unet_bpTR0_addsimmutate_addmoresimilar_finetune{epoch}.pt')
        if epoch > 0:
            #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_TR0_continuefrom99_{epoch}.pt')
            #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_RNAlign_restart_{epoch}.pt')
            #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_merge_alldata_{epoch}.pt')
            #torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/unet_train_on_TR0bpnewOriuseMXUnet_{epoch}.pt')
            torch.save(contact_net.state_dict(),  f'models_ckpt/final_model/for_servermodel/ufold_train_onalldata_{epoch}.pt')
        

#model_eval_all_test()
#torch.save(contact_net.module.state_dict(), model_path + 'unet_final.pt')
# sys.exit()








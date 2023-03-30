import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

# from FCN import FCNNet
from Network import U_Net as FCNNet
#from Network3 import U_Net_FP as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset
#from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections

args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess




def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict
# randomly select one sample from the test set and perform the evaluation

def model_eval_all_test(contact_net,test_generator):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    result_nc = list()
    result_nc_tmp = list()
    ct_dict_all = {}
    dot_file_dict = {}
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
    #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_merge_generator:
    #for contacts, seq_embeddings,seq_embeddings_1, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
        #pdb.set_trace()
        nc_map_nc = nc_map.float() * contacts
        if seq_lens.item() > 1500:
            continue
        if batch_n%1000==0:
            print('Batch number: ', batch_n)
        #if batch_n > 3:
        batch_n += 1
        #    break
        #if batch_n-1 in rep_ind:
        #    continue
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        # matrix_reps_batch = torch.unsqueeze(
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())
        #     torch.Tensor(matrix_reps.float()).to(device), -1)

        # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
        #     seq_len, seq_len]).to(device)

        # PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        tik = time.time()
        
        with torch.no_grad():
            #pred_contacts = contact_net(seq_embedding_batch,seq_embedding_batch_1)
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
            #seq_ori, 0.01, 0.1, 100, 1.6, True) ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()
        
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)
##      add fine tune
        #threshold = 0.5
        '''
        while map_no_train.sum(axis=1).max() > 1:
            u_no_train = postprocess(u_no_train,seq_ori, 0.01, 0.1, 50, 1.0, True)
            threshold += 0.005
            map_no_train = (u_no_train > threshold).float()
        '''
        ## end fine tune
        #pdb.set_trace()
        #ct_dict_all = get_ct_dict(map_no_train,batch_n,ct_dict_all)
        #ct_dict_all,dot_file_dict = get_ct_dict_fast(map_no_train,batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])
        #ct_dict_all,dot_file_dict = get_ct_dict_fast((contacts>0.5).float(),batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])
        #pdb.set_trace()
        
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        
        if nc_map_nc.sum() != 0:
            #pdb.set_trace()
            result_nc_tmp = list(map(lambda i: evaluate_exact_new(map_no_train_nc.cpu()[i],
                nc_map_nc.cpu().float()[i]), range(contacts_batch.shape[0])))
            result_nc += result_nc_tmp
            nc_name_list.append(seq_name[0])
            #if seq_lens.item() < 400 and result_nc_tmp[0][2] > 0.7:
                #pdb.set_trace()
            #    print(seq_name[0])
            #    print(result_no_train_tmp[0])
            #    print(result_nc_tmp[0])
        #seq_lens_list += list(seq_lens)

    #pdb.set_trace()
    #print(np.mean(run_time))
    
    #dot_ct_file = open('results/dot_ct_file.txt','w')
    '''
    dot_ct_file = open('results/nonredundant_true_dot_ct_file.txt','w')
    for i in range(batch_n):
        dot_ct_file.write('>%s\n'%(dot_file_dict[i+1][0][0]))
        dot_ct_file.write('%s\n'%(dot_file_dict[i+1][0][1]))
        #dot_ct_file.write('%s\n'%(dot_file_dict[i+1][0][2]))
        #dot_ct_file.write('\n')
    dot_ct_file.close()
    pdb.set_trace()
    '''
    '''
    ct_file = open('results/ct_file.txt','w')
    for i in range(batch_n):
        ct_file.write('>%d\n'%(i))
        for j in range(len(ct_dict_all[i+1])):
            ct_file.write('%d\t%d\n'%(ct_dict_all[i+1][j][0],ct_dict_all[i+1][j][1]))
        ct_file.write('\n')
    ct_file.close()
    '''
    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    #pdb.set_trace()
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))


    #with open('/data2/darren/experiment/ufold/results/sample_result.pickle','wb') as f:
    #    pickle.dump(result_dict,f)
    # with open('../results/rnastralign_short_pure_pp_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(0)
    
    #pdb.set_trace()
    
    config_file = args.config
    test_file = args.test_files
    
    config = process_config(config_file)
    #print('Here is the configuration of this run: ')
    #print(config)
    
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet30.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_reduce_noupscale25.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_reduce15.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_noupscale90.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_noupscale_simfam90.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_upscale7fam_90.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_ori49.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_8dim49.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar17.pt' #unet_bpTR0_addsimmutate_addmoresimilar48.pt
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0_33.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_RNAlign_49.pt'
    if test_file not in ['TS1','TS2','TS3']:
        MODEL_SAVED = 'models/ufold_train.pt'    
    else:
        MODEL_SAVED = 'models/ufold_train_pdbfinetune.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0andMXUnet_99.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0bpnewOriuseMXUnet_96.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0_extract_99.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/final_model/unet_train_on_TR0_continuefrom99_56.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar_finetune0.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar_twochannel48.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpTR0_addsimmutate_addmoresimilar_25dim46.pt'
    #MODEL_SAVED = '/data2/darren/experiment/models_ckpt/concat_trainsetA/unet_trainsetA14.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_upscaleonlyPDB_outer7.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_rfam12_upscaleallfams_19.ptunet_rfam12_upscaleonlyPDB_finetune8.pt'
    #MODEL_SAVED = '/data2/darren/experiment/models_ckpt/concat/'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet_bpRNA26.pt'
    #MODEL_SAVED = '/data2/darren/experiment/ufold/models_ckpt/unet48.pt'
    
    # os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
    
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    model_path = '/data2/darren/experiment/ufold/models_ckpt/'.format(model_type, data_type,d)
    epoches_first = config.epoches_first
    
    
    # if gpu is to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    seed_torch()
    
    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
    
    # train_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'train', True)
    # val_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'val')
    ##test_data = RNASSDataGenerator('./data/{}/'.format(data_type), 'test_no_redundant.pickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant.pickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant_600.pickle')
    print('Loading test file: ',test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file+'.pickle')
    else:
        test_data = RNASSDataGenerator('data/',test_file+'.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'all_1800_archieveII.pickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'all_600_archieveII.pickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_test_generate.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_RF00001_similarfamilys_test.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TestSetA.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_allfamily_generate_test.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA12_38family_generate_test.cPickle')
    ##test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_new_generate.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_new20201015.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'archieveII_contacts_pred.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_pdbnewgenerate_yingxc.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TS0_ori.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TR0_leavefortest1000.cPickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/', 'bpRNA_TR0_andsim_mutate_extract_train.cPickle')
    seq_len = test_data.data_y.shape[-2]
    print('Max seq length ', seq_len)
    #pdb.set_trace()
    
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    # # train_set = Dataset(train_data)
    # train_set = Dataset_FCN(train_data)
    # train_generator = data.DataLoader(train_set, **params)
    
    # # val_set = Dataset(val_data)
    # val_set = Dataset_FCN(val_data)
    # val_generator = data.DataLoader(val_set, **params)
    
    # test_set = Dataset(test_data)
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)
    
    '''
    test_merge = Dataset_FCN_merge(test_data,test_data2)
    test_merge_generator = data.DataLoader(test_merge, **params)
    pdb.set_trace()
    '''
    
    
    
    contact_net = FCNNet(img_ch=17)
    
    #pdb.set_trace()
    print('==========Start Loading==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cuda:1'))
    print('==========Finish Loading==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    model_eval_all_test(contact_net,test_generator)
    
    
    
    # if LOAD_MODEL and os.path.isfile(model_path):
    #     print('Loading u net model...')
    #     contact_net.load_state_dict(torch.load(model_path))
    
    
    # u_optimizer = optim.Adam(contact_net.parameters())
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()







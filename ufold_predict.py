import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

# from FCN import FCNNet
from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset,RNASSDataGenerator_input
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections

import subprocess
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
    
'''
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in torch.nonzero(seq_embedding,as_tuple=False)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict
# randomly select one sample from the test set and perform the evaluation
'''
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    #pdb.set_trace()
    #print(seq_name)
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmpp = np.copy(seq_tmp)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    letter='AUCG'
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    #seq = ((seq_tmp+1).squeeze()[:len(seq_letter)],torch.arange(predict_matrix.shape[-1]).numpy()[:len(seq_letter)]+1)
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    dot_file_dict[batch_num] = [(seq_name.replace('/','_'),seq_letter,dot_list[:len(seq_letter)])]
    #pdb.set_trace()
    ct_file_output(ct_dict[batch_num],seq_letter,seq_name,'results/save_ct_file')
    _,_,noncanonical_pairs = type_pairs(ct_dict[batch_num],seq_letter)
    tertiary_bp = [list(x) for x in set(tuple(x) for x in noncanonical_pairs)]
    str_tertiary = []
    for i,I in enumerate(tertiary_bp):
        if i==0:
            str_tertiary += ('(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')
        else:
            str_tertiary += (';(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')

    tertiary_bp = ''.join(str_tertiary)
    #return ct_dict,dot_file_dict
    return ct_dict,dot_file_dict,tertiary_bp

def ct_file_output(pairs, seq, seq_name, save_result_path):

    #pdb.set_trace()
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]-1] = int(I[1])
        #col5[I[1]] = int(I[0]) + 1
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    #os.chdir(save_result_path)
    #print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(os.path.join(save_result_path, seq_name.replace('/','_'))+'.ct', (temp), delimiter='\t', fmt="%s", header='>seq length: ' + str(len(seq)) + '\t seq name: ' + seq_name.replace('/','_') , comments='')

    return

def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    # seq_pairs = [[sequence[i[0]],sequence[i[1]]] for i in pairs]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0]-1],sequence[i[1]-1]] in [["A","U"], ["U","A"]]:
            AU_pair.append(i)
        elif [sequence[i[0]-1],sequence[i[1]-1]] in [["G","C"], ["C","G"]]:
            GC_pair.append(i)
        elif [sequence[i[0]-1],sequence[i[1]-1]] in [["G","U"], ["U","G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
        # print(watson_pairs_t, wobble_pairs_t, other_pairs_t)
    return watson_pairs_t, wobble_pairs_t, other_pairs_t


def model_eval_all_test(contact_net,test_generator):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    seq_names = []
    ct_dict_all = {}
    dot_file_dict = {}
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    for seq_embeddings, seq_lens, seq_ori, seq_name in test_generator:
    #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        if batch_n%100==0:
            print('Sequencing number: ', batch_n)
        #pdb.set_trace()
        #if batch_n > 3:
        #    break
        batch_n += 1
        #if batch_n-1 in rep_ind:
        #    continue
        #contacts_batch = torch.Tensor(contacts.float()).to(device)
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
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
            #seq_ori, 0.01, 0.1, 50, 1, True)
        map_no_train = (u_no_train > 0.5).float()
        #pdb.set_trace()
        threshold = 0.5
        th = 0
        '''
        while map_no_train.sum(axis=1).max() > 1:
            #u_no_train = postprocess(u_no_train,seq_ori, 0.01, 0.1, 50, 1.0, True)
            #pdb.set_trace()
            threshold += 0.01
            #print(th)
            map_no_train = (u_no_train > threshold).float()
        '''
        #ct_dict_all = get_ct_dict(map_no_train,batch_n,ct_dict_all)
        if seq_name[0].startswith('.'):
            seq_name = [seq_name[0][1:]]
        seq_names.append(seq_name[0].replace('/','_'))
        #ct_dict_all,dot_file_dict = get_ct_dict_fast(map_no_train,batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])
        ct_dict_all,dot_file_dict,tertiary_bp = get_ct_dict_fast(map_no_train,batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])
        #ct_dict_all,dot_file_dict = get_ct_dict_fast((contacts>0.5).float(),batch_n,ct_dict_all,dot_file_dict,seq_ori.cpu().squeeze(),seq_name[0])
        ## draw plot section
        if not args.nc:
            subprocess.Popen(["java", "-cp", "VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', 'results/save_ct_file/' + seq_name[0].replace('/','_') + '.ct', '-o', 'results/save_varna_fig/' + seq_name[0].replace('/','_') + '_radiate.png', '-algorithm', 'radiate', '-resolution', '8.0', '-bpStyle', 'lw'], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        else:
            subprocess.Popen(["java", "-cp", "VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', 'results/save_ct_file/' + seq_name[0].replace('/','_') + '.ct', '-o', 'results/save_varna_fig/' + seq_name[0].replace('/','_') + '_radiatenew.png', '-algorithm', 'radiate', '-resolution', '8.0', '-bpStyle', 'lw','-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        #subprocess.Popen(["java", "-cp", "VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', 'results/save_ct_file/' + seq_name[0].replace('/','_') + '.ct', '-o', 'results/save_varna_fig/' + seq_name[0].replace('/','_') + '_radiate_ground_truth.png', '-algorithm', 'radiate', '-resolution', '8.0', '-bpStyle', 'lw'], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        #subprocess.Popen(["java", "-cp", "VARNAv3-93.jar", "fr.orsay.lri.varna.applications.VARNAcmd", '-i', 'results/save_ct_file/' + seq_name[0].replace('/','_') + '.ct', '-o', 'results/save_varna_fig/' + seq_name[0].replace('/','_') + '_radiate_ground_truthnew.png', '-algorithm', 'naview', '-resolution', '18.0', '-bpStyle', 'lw','-auxBPs', tertiary_bp], stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0]
        #pdb.set_trace()
        '''
        result_no_train_tmp = list(map(lambda i: evaluate_exact(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        result_no_train_tmp_shift = list(map(lambda i: evaluate_shifted(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train_shift += result_no_train_tmp_shift
        '''
        seq_lens_list += list(seq_lens)

    #pdb.set_trace()
    
    ct_file_name_list = ['results/save_ct_file/'+item+'.ct' for item in seq_names]
    subprocess.getstatusoutput('sed -s \'$G\' '+' '.join(ct_file_name_list)+' > results/save_ct_file/ct_file_merge.ct')
    #dot_ct_file = open('results/dot_ct_file.txt','w')
    dot_ct_file = open('results/input_dot_ct_file.txt','w')
    for i in range(batch_n):
        dot_ct_file.write('>%s\n'%(dot_file_dict[i+1][0][0]))
        dot_ct_file.write('%s\n'%(dot_file_dict[i+1][0][1]))
        dot_ct_file.write('%s\n'%(dot_file_dict[i+1][0][2]))
        dot_ct_file.write('\n')
    dot_ct_file.close()
    '''
    ct_file = open('results/ct_file.txt','w')
    for i in range(batch_n):
        ct_file.write('>%d\n'%(i))
        for j in range(len(ct_dict_all[i+1])):
            ct_file.write('%d\t%d\n'%(ct_dict_all[i+1][j][0],ct_dict_all[i+1][j][1]))
        ct_file.write('\n')
    ct_file.close()
    '''
    '''
    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    nt_shift_p,nt_shift_r,nt_shift_f1 = zip(*result_no_train_shift)  
    #pdb.set_trace()
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
    '''

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(1)

    print('Welcome using UFold prediction tool!!!')

    if not os.path.exists('results/save_ct_file'):
        os.makedirs('results/save_ct_file')
    if not os.path.exists('results/save_varna_fig'):
        os.makedirs('results/save_varna_fig')
    config_file = args.config
    test_file = args.test_files

    config = process_config(config_file)
    
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    #model_path = '/data2/darren/experiment/ufold/models_ckpt/'.format(model_type, data_type,d)
    epoches_first = config.epoches_first

    MODEL_SAVED = 'models/unet_train_on_merge_alldata_98.pt'

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    seed_torch()
        
    test_data = RNASSDataGenerator_input('data/', 'input')
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}

    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)
    contact_net = FCNNet(img_ch=17)

    #pdb.set_trace()
    print('==========Start Loading Pretrained Model==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cuda:1'))
    print('==========Finish Loading Pretrained Model==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    model_eval_all_test(contact_net,test_generator)
    print('==========Done!!! Please check results folder for the predictions!==========')

    
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()






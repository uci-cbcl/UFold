import collections
import subprocess

from torch.utils import data

from Network import U_Net as FCNNet
from ufold.config import process_config
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from ufold.data_generator import RNASSDataGenerator_input
from ufold.utils import *

args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess


def get_device():
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda:0")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


def get_seq(contact):
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis=1).clamp_max(1))
    seq[contact.sum(axis=1) == 0] = -1
    return seq


def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(["_"] * len(seq))
    dot_file[seq > idx] = "("
    dot_file[seq < idx] = ")"
    dot_file[seq == 0] = "."
    dot_file = "".join(dot_file)
    return dot_file


def get_ct_dict(predict_matrix, batch_num, ct_dict):

    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:, i, j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i, j)]
                else:
                    ct_dict[batch_num] = [(i, j)]
    return ct_dict


def get_ct_dict_fast(predict_matrix, batch_num, ct_dict, dot_file_dict, seq_embedding, seq_name):
    seq_tmp = (
        torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis=1).clamp_max(1))
        .numpy()
        .astype(int)
    )
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1
    dot_list = seq2dot((seq_tmp + 1).squeeze())
    letter = "AUCG"
    seq_letter = "".join([letter[item] for item in np.nonzero(seq_embedding)[:, 1]])
    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    ct_dict[batch_num] = [(seq[0][i], seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    dot_file_dict[batch_num] = [(seq_name.replace("/", "_"), seq_letter, dot_list[: len(seq_letter)])]
    ct_file_output(ct_dict[batch_num], seq_letter, seq_name, "results/save_ct_file")
    _, _, noncanonical_pairs = type_pairs(ct_dict[batch_num], seq_letter)
    tertiary_bp = [list(x) for x in set(tuple(x) for x in noncanonical_pairs)]
    str_tertiary = []
    for i, I in enumerate(tertiary_bp):
        if i == 0:
            str_tertiary += "(" + str(I[0]) + "," + str(I[1]) + '):color=""#FFFF00""'
        else:
            str_tertiary += ";(" + str(I[0]) + "," + str(I[1]) + '):color=""#FFFF00""'

    tertiary_bp = "".join(str_tertiary)
    return ct_dict, dot_file_dict, tertiary_bp


def ct_file_output(pairs, seq, seq_name, save_result_path):
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0] - 1] = int(I[1])
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack(
        (
            np.char.mod("%d", col1),
            col2,
            np.char.mod("%d", col3),
            np.char.mod("%d", col4),
            np.char.mod("%d", col5),
            np.char.mod("%d", col6),
        )
    ).T
    np.savetxt(
        os.path.join(save_result_path, seq_name.replace("/", "_")) + ".ct",
        (temp),
        delimiter="\t",
        fmt="%s",
        header=">seq length: " + str(len(seq)) + "\t seq name: " + seq_name.replace("/", "_"),
        comments="",
    )

    return


def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0] - 1], sequence[i[1] - 1]] in [["A", "U"], ["U", "A"]]:
            AU_pair.append(i)
        elif [sequence[i[0] - 1], sequence[i[1] - 1]] in [["G", "C"], ["C", "G"]]:
            GC_pair.append(i)
        elif [sequence[i[0] - 1], sequence[i[1] - 1]] in [["G", "U"], ["U", "G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
    return watson_pairs_t, wobble_pairs_t, other_pairs_t


def model_eval_all_test(contact_net, test_generator):
    device = get_device()
    contact_net.to(device)
    contact_net.train()
    seq_lens_list = list()
    batch_n = 0
    seq_names = []
    ct_dict_all = {}
    dot_file_dict = {}
    for seq_embeddings, seq_lens, seq_ori, seq_name in test_generator:
        if batch_n % 100 == 0:
            print("Sequencing number: ", batch_n)
        batch_n += 1
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
        map_no_train = (u_no_train > 0.5).float()
        if seq_name[0].startswith("."):
            seq_name = [seq_name[0][1:]]
        seq_names.append(seq_name[0].replace("/", "_"))
        ct_dict_all, dot_file_dict, tertiary_bp = get_ct_dict_fast(
            map_no_train, batch_n, ct_dict_all, dot_file_dict, seq_ori.cpu().squeeze(), seq_name[0]
        )
        ## draw plot section
        # TODO(john): reenable plotting
        # if not args.nc:
        #     subprocess.Popen(
        #         [
        #             "java",
        #             "-cp",
        #             "VARNAv3-93.jar",
        #             "fr.orsay.lri.varna.applications.VARNAcmd",
        #             "-i",
        #             "results/save_ct_file/" + seq_name[0].replace("/", "_") + ".ct",
        #             "-o",
        #             "results/save_varna_fig/" + seq_name[0].replace("/", "_") + "_radiate.png",
        #             "-algorithm",
        #             "radiate",
        #             "-resolution",
        #             "8.0",
        #             "-bpStyle",
        #             "lw",
        #         ],
        #         stderr=subprocess.STDOUT,
        #         stdout=subprocess.PIPE,
        #     ).communicate()[0]
        # else:
        #     subprocess.Popen(
        #         [
        #             "java",
        #             "-cp",
        #             "VARNAv3-93.jar",
        #             "fr.orsay.lri.varna.applications.VARNAcmd",
        #             "-i",
        #             "results/save_ct_file/" + seq_name[0].replace("/", "_") + ".ct",
        #             "-o",
        #             "results/save_varna_fig/" + seq_name[0].replace("/", "_") + "_radiatenew.png",
        #             "-algorithm",
        #             "radiate",
        #             "-resolution",
        #             "8.0",
        #             "-bpStyle",
        #             "lw",
        #             "-auxBPs",
        #             tertiary_bp,
        #         ],
        #         stderr=subprocess.STDOUT,
        #         stdout=subprocess.PIPE,
        #     ).communicate()[0]
        seq_lens_list += list(seq_lens)

    ct_file_name_list = ["results/save_ct_file/" + item + ".ct" for item in seq_names]
    subprocess.getstatusoutput(
        "sed -s '$G' " + " ".join(ct_file_name_list) + " > results/save_ct_file/ct_file_merge.ct"
    )
    dot_ct_file = open("results/input_dot_ct_file.txt", "w")
    for i in range(batch_n):
        dot_ct_file.write(">%s\n" % (dot_file_dict[i + 1][0][0]))
        dot_ct_file.write("%s\n" % (dot_file_dict[i + 1][0][1]))
        dot_ct_file.write("%s\n" % (dot_file_dict[i + 1][0][2]))
        dot_ct_file.write("\n")
    dot_ct_file.close()


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    # torch.cuda.set_device(1)

    print("Welcome using UFold prediction tool!!!")

    if not os.path.exists("results/save_ct_file"):
        os.makedirs("results/save_ct_file")
    if not os.path.exists("results/save_varna_fig"):
        os.makedirs("results/save_varna_fig")
    config_file = args.config
    test_file = args.test_files

    config = process_config(config_file)

    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    epoches_first = config.epoches_first

    MODEL_SAVED = "models/ufold_train_alldata.pt"

    device = get_device()

    seed_torch()

    # TODO(john): looks like sequences >600 bps are truncated to 600
    test_data = RNASSDataGenerator_input("data/", "input")

    params = {"batch_size": BATCH_SIZE, "shuffle": True, "num_workers": 4, "drop_last": True}

    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)
    contact_net = FCNNet(img_ch=17)

    print("==========Start Loading Pretrained Model==========")
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))
    print("==========Finish Loading Pretrained Model==========")
    contact_net.to(device)
    model_eval_all_test(contact_net, test_generator)
    print("==========Done!!! Please check results folder for the predictions!==========")


if __name__ == "__main__":
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple("RNA_SS_data", "seq ss_label length name pairs")
    main()

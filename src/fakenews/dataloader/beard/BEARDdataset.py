import torch, os, pickle
from torch.utils.data import Dataset


'''
EDataset, used by BEARD specifically.
It seems that the argument "max_psot_len" is unused anywhere else in the original codebase.
We temporaily leave it here, and if we assure that this hyperparameter does not do anything, it will be removed.
'''
class BEARDataset(Dataset):
    def __init__(self, data, interval, max_seq_len, max_post_len, device, num_events):
        self.device = device
        self.num_events = num_events

        '''
        The time unit converter. The unit of the original timestamp is in seconds.
        You can assign interval = 3600 in the dataloader config to change the time unit to hours.
        '''
        self.interval = interval
        '''
        How many tweets at most are available for fake news detection models to judge the veracity of a topic?
        This parameter is super helpful when you conduct experiments about early rumor detection tasks.
        '''
        self.max_seq_len = max_seq_len
        '''
        unknown parameter.
        '''
        self.max_post_len = max_post_len
        
        '''
        BEARDataset does not deal with folding anymore.
        '''
        self.data = data
        self.index_to_eid = list(self.data.keys())


    def __len__(self):
        return len(self.data)
    

    def __label_convert__(self,label):
        if label == 1:
            target = [0,1]
        else:
            target = [1,0]
        return target
    

    def __time_convert__(self,merge_times):

        start_timestamp = merge_times[0][-1]
        time_last_arrival = [0.0]
        time_since_starts = []

        for x in merge_times:
            tmp = (x[-1] - start_timestamp)/self.interval
            assert tmp >= 0
            time_since_starts.append(tmp)

        for x in zip(time_since_starts[1:],time_since_starts[:-1]):
            tmp = x[0] - x[1]
            assert tmp >= 0
            time_last_arrival.append(tmp)

        return time_since_starts,time_last_arrival
    

    def __index_convert__(self,merge_tids):
        last_post_index = [0]*len(merge_tids)
        for i,seq in enumerate(merge_tids):
            last_post_index[i] = len(seq) + last_post_index[i-1]
        return last_post_index


    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            eid = self.index_to_eid[index]
            seqs = self.data[eid]['merge_seqs']
            label = self.data[eid]['label']
    
            all_lens = list(map(len, seqs['merge_times']))
            seq_length = sum(all_lens)
    
            merge_times = seqs['merge_times'][:self.max_seq_len]
            merge_tids = seqs['merge_tids'][:self.max_seq_len]
            text_vec_seq_t = seqs['merge_vecs'][:self.max_seq_len]
            try:
                time_since_starts,time_last_arrival = self.__time_convert__(merge_times[:self.max_seq_len])
            except Exception as e:
                print(eid)
                print(merge_times)
                print(e)
    
            last_post_index = self.__index_convert__(merge_tids)
            target = self.__label_convert__(int(label))
            text_vec_seq= text_vec_seq_t
            return label,target,text_vec_seq,last_post_index,time_since_starts,time_last_arrival,seq_length,eid,merge_tids
    

    def __call__(self, batch_data):
        sorted_batch = sorted(batch_data, key = lambda x: len(x[2]), reverse=True)
        eids = [seq[7] for seq in sorted_batch]
        label_seqs = torch.LongTensor([int(seq[0]) for seq in sorted_batch])
        target_seqs = torch.LongTensor([seq[1] for seq in sorted_batch])
        text_seqs = [seq[2] for seq in sorted_batch]
        seqs_length = torch.LongTensor(list(map(len, text_seqs)))
        real_lengths = torch.LongTensor([seq[6] for seq in sorted_batch])
        post_since_start_seqs = [seq[3] for seq in sorted_batch]
        last_arrival_time_seqs = [seq[5] for seq in sorted_batch]
        times_since_start_seqs = [seq[4] for seq in sorted_batch]
        all_tids = [seq[8] for seq in sorted_batch]
        end_time_seqs = torch.FloatTensor([seq[-1] for seq in times_since_start_seqs])
        posts_length,max_post_len = [[] for x in text_seqs],None

        d_input_text_embeddings = len(text_seqs[0][0])
        text_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max(), d_input_text_embeddings).float()

        arrival_time_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
        times_since_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()
        post_since_start_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
        posts_length_tensor = torch.ones(len(sorted_batch), seqs_length.max()).long()

        for idx, (text_seq, time_seq, seqlen,timstamp_seq,index_seq,posts_len) in enumerate(zip(text_seqs, last_arrival_time_seqs, seqs_length,times_since_start_seqs,post_since_start_seqs,posts_length)):

            text_seqs_tensor[idx, :seqlen,:] = torch.FloatTensor(text_seq)

            arrival_time_seqs_tensor[idx, :seqlen] = torch.FloatTensor(time_seq)
            times_since_seqs_tensor[idx,:seqlen] = torch.FloatTensor(timstamp_seq)
            post_since_start_seqs_tensor[idx,:seqlen] = torch.LongTensor(index_seq)
 
        return label_seqs, target_seqs, text_seqs_tensor, post_since_start_seqs_tensor, \
               arrival_time_seqs_tensor, end_time_seqs, seqs_length, times_since_seqs_tensor, posts_length_tensor, \
               max_post_len, real_lengths,eids, all_tids


def read_data(data_path, file_names):
    # Here we load the dataset into a dict
    data_raw = {}
    try:
        for file_name in file_names:
            file, _ = file_name.split('.')
            dataset_full_filepath = os.path.join(data_path, file_name)
            f = open(dataset_full_filepath, 'rb')
            data_raw[file] = pickle.load(f)
            f.close()
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {data_path}")
    
    return data_raw


def get_dataset_for_beard():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [BEARDataset, read_data]
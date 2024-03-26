import torch
import torchaudio
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader


char2idx = {
    'z': 0,
    **{str(i): i for i in range(1, 10)},
    'o': 10
}


def label_from_name(name):
    file_name = name.split('/')[-1]
    digits = file_name.replace('.wav', '')[:-1] # remove ,wav and last a/b
    label = [char2idx[d] for d in digits]
    return label

def collate_fn_train(batch):
    labels = [torch.tensor(x[1]) for x in batch]
    target_lengths = [len(x) for x in labels]
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

    spectograms = [x[0].squeeze(0).T for x in batch]
    input_lengths = [s.size(0) for s in spectograms]
    spectograms = nn.utils.rnn.pad_sequence(spectograms, batch_first=True)
    return spectograms, labels, input_lengths, target_lengths

class AudioDataset(Dataset):
    def __init__(self, split):
        self.split = split
        if split == 'test':
            self.files = [f'ex3/test/{f}' for f in os.listdir('ex3/test')]
        else:
            self.files = []
            for age in 'adults', 'children':
                for gender in (['boy', 'girl'] if age == 'children' else ['man', 'woman']):
                    all_names = os.listdir(f'ex3/train/{age}/{gender}')
                    all_names = [name for name in all_names if name != '.DS_Store']
                    for name in all_names:
                        self.files.extend([f'ex3/train/{age}/{gender}/{name}/{f}' for f in os.listdir(f'ex3/train/{age}/{gender}/{name}') if f.endswith('.wav')])
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.files[idx])
        spectogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
        if self.split == 'test':
            return spectogram
        return spectogram, label_from_name(self.files[idx])


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

if __name__ == '__main__':
    dataset = AudioDataset('train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_train)
    for _ in dataloader:
        pass
    
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#import matplotlib
##%matplotlib inline
#import matplotlib.pylab as plt

import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from pydub import AudioSegment
import gdown

def plot_data(data, figsize=(16, 4)):
        fig, axes = plt.subplots(1, len(data), figsize=figsize)
        for i in range(len(data)):
            axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                           interpolation='none')

def create_model(metin):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    
#    model1_output = "burakAskin2.pt"
    model1_url = "https://drive.google.com/uc?id=1-BwU-F6mEbenU8awgdE-sRKtrLEOatU_"
    model1_output = "tacotron2.pt"
    gdown.download(model1_url, model1_output, quiet=False)
    model = load_model(hparams)
    model.load_state_dict(torch.load(model1_output)['state_dict'])
    _ = model.cuda().eval().half()
    
    model2_url = "https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
    model2_output = "waveglow.pt"
    gdown.download(model2_url, model2_output, quiet=False)
#    waveglow_path = 'waveglow_256channels_universal_v5.pt'
#    model2_output = 'waveglow_256channels_universal_v5.pt' #eklendi
#    model2 = torch.load(waveglow_path) #eklendi
#    torch.save(model2, model2_output) #eklendi
    waveglow = torch.load(model2_output)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
#    denoiser = Denoiser(waveglow)
    
    text = metin
    sequence = np.array(text_to_sequence(text, ['turkish_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
    
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.85)     # sigma=1
        audio_path = './static/a.wav'
        audio = ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
        audio = AudioSegment(audio.data, frame_rate=22050, sample_width=2, channels=1)
        audio.export(audio_path, format="wav", bitrate="64k")   
    
    return audio
        

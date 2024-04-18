import librosa
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchcrepe
import gradio as gr
import matplotlib as mpl
from multiprocessing import Process

def main(audio_path,target_f0,length,hop_size):
    if audio_path in [None,'']:
        return None,'audio?'
    y,sr=librosa.load(audio_path,sr=None)
    audio = torch.tensor(np.copy(y))[None]

    fmin = 50
    fmax = 550
    model = 'tiny'
    batch_size = 2048   
    hop_length = int(sr*hop_size/ 1000)
    pitch = torchcrepe.predict(audio,
                           sr,
                           hop_length,
                           fmin,
                           fmax,
                           model,
                           batch_size=batch_size,
                           device='cuda:0'
                        )
    #print(pitch)
    values_list = pitch.tolist()[0]
    f0=sum(values_list)/len(values_list)
    print(f0)
    ##resample
    tr_sr = int(sr * target_f0 / f0)
    y = librosa.resample(y, orig_sr=tr_sr, target_sr=sr)
    ##time stretch
    y=librosa.effects.time_stretch(y, rate=(sr/tr_sr)/length)
    sf.write("output_audio.wav", y, sr)
    return (sr,y),'success'

def plot(audio_path):
    if audio_path in [None, ""]:
        return 'audio?'
    '''
    y,sr=librosa.load(audio_path,sr=None)   
    m=librosa.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect')
    '''
    p=Process(target=plot_show, args=[audio_path])
    p.start()
    p.join()
    return 'ok'

def plot_show(audio_path):
    sr, y = wavfile.read(audio_path)
    m = np.fft.fft(y)
    plt.figure()
    plt.plot(np.abs(m))
    mpl.use("TkAgg")
    plt.show()    

if __name__=="__main__":
    with gr.Blocks(title="VC") as app:
        with gr.Tabs():
            with gr.TabItem(label=''):
                with gr.Row():
                    with gr.Column():
                        target_f0=gr.Slider(label="目标频率",minimum=50,maximum=500,step=10,value=180)
                        length_scale = gr.Slider(label="音频播放长度",minimum=0.5, maximum=2, step=0.1,value=1)
                        hop_size = gr.Slider(label="hop_size(ms)", minimum=5, maximum=200, step=5,value=10)
                    with gr.Column():
                        input_audio=gr.Audio(label="输入音频",type='filepath')
                        output_audio=gr.Audio(label="输出音频")
                        textbox=gr.Textbox(label='',interactive=False,value='')
                        vc_btn=gr.Button(value="开始转换",variant='primary')
                        plot_btn=gr.Button(value='画图',variant='secondary')
        vc_btn.click(fn=main,inputs=[input_audio,target_f0,length_scale,hop_size],outputs=[output_audio,textbox])
        plot_btn.click(fn=plot,inputs=[input_audio],outputs=[textbox])
    app.launch(share=False,inbrowser=True)
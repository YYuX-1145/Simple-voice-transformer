import librosa
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchcrepe
import os
import gradio as gr


# audio_name='audio2.wav'
# TARGET_F0=150


def main(audio_path,target_f0,length):
    if audio_path in [None,'']:
        return None,'audio?'
    y,sr=librosa.load(audio_path,sr=None)
    audio = torch.tensor(np.copy(y))[None]

    fmin = 50
    fmax = 550
    model = 'tiny'
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 2048
    hop_length = int(sr / 100.0)
    # Compute pitch using first gpu
    pitch = torchcrepe.predict(audio,
                           sr,
                           hop_length,
                           fmin,
                           fmax,
                           model,
                           batch_size=batch_size,
                           device='cuda:0'
                        )
    print(pitch)
    values_list = pitch.tolist()[0]
    f0=0.0
    for s,i in enumerate(values_list):
        f0+=i
    f0=f0/s
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
    sr, y = wavfile.read(audio_path)
    m = np.fft.fft(y)
    plt.figure()
    plt.plot(np.abs(m))
    plt.savefig('fig.png',dpi=300)
    os.startfile("fig.png")
    return 'ok'


if __name__=="__main__":
    with gr.Blocks(title="VC") as app:
        with gr.Tabs():
            with gr.TabItem(label=''):
                with gr.Row():
                    with gr.Column():
                        target_f0=gr.Slider(label="目标频率",minimum=50,maximum=500,step=1,value=180)
                        length_scale = gr.Slider(label="音频播放长度",minimum=0.5, maximum=2, step=0.1,value=1)
                    with gr.Column():
                        input_audio=gr.Audio(label="输入音频",type='filepath')
                        output_audio=gr.Audio(label="输出音频")
                        textbox=gr.Textbox(label='',interactive=False,value='')
                        vc_btn=gr.Button(value="开始转换")
                        plot_btn=gr.Button(value='画图')
        vc_btn.click(fn=main,inputs=[input_audio,target_f0,length_scale],outputs=[output_audio,textbox])
        plot_btn.click(fn=plot,inputs=[input_audio],outputs=[textbox])
    app.launch(share=False,inbrowser=True)

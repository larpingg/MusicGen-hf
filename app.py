

from tempfile import NamedTemporaryFile
import torch
import gradio as gr
from audiocraft.models import MusicGen

from audiocraft.data.audio import audio_write


MODEL = None


def load_model(version):
    print("loading..", version)
    return MusicGen.get_pretrained(version)


def predict(model, text, melody, duration, topk, topp, temperature, cfg_coef):
    global MODEL
    topk = int(topk)
    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)

    if duration > MODEL.lm.cfg.dataset.segment_duration:
        raise gr.Error("erorr")
    MODEL.set_generation_params(
        use_sampling=True,
        top_k=topk,
        top_p=topp,
        temperature=temperature,
        cfg_coef=cfg_coef,
        duration=duration,
    )

    if melody:
        sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
        print(melody.shape)
        if melody.dim() == 2:
            melody = melody[None]
        melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
        output = MODEL.generate_with_chroma(
            descriptions=[text],
            melody_wavs=melody,
            melody_sample_rate=sr,
            progress=False
        )
    else:
        output = MODEL.generate(descriptions=[text], progress=False)

    output = output.detach().cpu().float()[0]
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        audio_write(file.name, output, MODEL.sample_rate, strategy="loudness", add_suffix=False)
        waveform_video = gr.make_waveform(file.name)
    return waveform_video


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # stllcld gui
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                text = gr.Text(label="input prompt", interactive=True)
                melody = gr.Audio(source="upload", type="numpy", label="reference/input", interactive=True)
            with gr.Row():
                submit = gr.Button("submit")
            with gr.Row():
                model = gr.Radio(["stllcld 1", "stllcld medium", "stllcld small", "stllcld large"], label="Model", value="melody", interactive=True)
            with gr.Row():
                duration = gr.Slider(minimum=1, maximum=10000, value=10, label="duration", interactive=True)
            with gr.Row():
                topk = gr.Number(label="top k", value=250, interactive=True)
                topp = gr.Number(label="top p", value=0, interactive=True)
                temperature = gr.Number(label="temp", value=1.0, interactive=True)
                cfg_coef = gr.Number(label="classifier", value=3.0, interactive=True)
        with gr.Column():
            output = gr.Video(label="output")
    submit.click(predict, inputs=[model, text, melody, duration, topk, topp, temperature, cfg_coef], outputs=[output])
    gr.Examples(
        fn=predict,
        examples=[
            
            [
                "lofi slow bpm electro chill with organic samples",
                None,
                "medium",
            ],
        ],
        inputs=[text, melody, model],
        outputs=[output]
    )
    
    gr.Markdown(
        """
        ## dm @talenaatfield to donate
        """
    )

demo.launch()

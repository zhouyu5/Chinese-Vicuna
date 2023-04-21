import time
import gradio as gr
# from utils import SteamGenerationMixin
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

### Vicuna-7B Model ###
tokenizer = AutoTokenizer.from_pretrained("/home/vmagent/app/data/vicuna-7b")
model = AutoModelForCausalLM.from_pretrained("/home/vmagent/app/data/vicuna-7b", low_cpu_mem_usage=True)
model.eval()

### Alpaca-13B Model ###
#tokenizer = LlamaTokenizer.from_pretrained("/mnt/DP_disk1/dataset/alpaca-13b/")
#model = AutoModelForCausalLM.from_pretrained("/mnt/DP_disk1/dataset/alpaca-13b/")
#model.eval()

### Prompt Format Generation ###
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

### Vicuna Chat ###
def chat(message, history):
    prompt = generate_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model.generate(input_ids=input_ids,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=512,
        do_sample=True)
    s = outputs.sequences[0]
    output = tokenizer.decode(s)
    response = output.split("### Response:")[1].strip().split('###')[0]
    history.append((message, response))
    return "", history 


### Gradio Component Definition ###
with gr.Blocks() as demo:
    title = gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>"
        + "Vicuna Chatbot of Intel® End-to-End AI Optimization Kit"
        + "</h1>"
        )
    description = gr.Markdown(
        """
        This demo was used to demonstrate the benefit of [Intel® End-to-End AI Optimization Kit](https://github.com/intel/e2eAIOK) on the large language model vicuna (7B).

        Try it and chat with vicuna chatbot! Please note that the demo is in preview under limited HW resources. We are committed to continue improving the demo and happy to hear your feedbacks. Thanks for your trying!
        """
    )

    with gr.Column(variant="panel"):
        chatbot = gr.Chatbot().style(height=512)

        with gr.Row():
            msg_input = gr.components.Textbox(label="Input")
            clear_button = gr.Button("Clear").style(full_width=False, size='sm')

        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            scroll_to_output=True,
        )
        clear_button.click(lambda: (None, None), None, chatbot, queue=False)

demo.launch(share=True, server_name="0.0.0.0", server_port=7861)
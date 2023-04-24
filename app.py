import sys
import torch
import transformers
import json
import gradio as gr
import argparse
import warnings
import os
from utils import SteamGenerationMixin, printf
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizer, TextIteratorStreamer
from threading import Thread

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


parser = argparse.ArgumentParser()
parser.add_argument("--model_path_list", type=str, default="EleutherAI/gpt-neo-2.7B")
parser.add_argument("--lora_path_list", type=str, default="")
parser.add_argument("--use_typewriter", type=int, default=1)
parser.add_argument("--share_link", type=int, default=1)
parser.add_argument("--use_local", type=int, default=0)
parser.add_argument("--load_8bit_list", type=str, default='0')
args = parser.parse_args()

load_8bit_list = [bool(int(item)) for item in args.load_8bit_list.split(',')]
model_path_list = args.model_path_list.split(',')
lora_path_list = args.lora_path_list.split(',') if ',' in args.lora_path_list else ['']*len(model_path_list)
model_name_list = [model_path.split('/')[-1] for model_path in model_path_list]
model_name_dict = {
    model_name: i
    for i, model_name in enumerate(model_name_list)
} 


def load_model_and_tokenizer(model_path, lora_path, LOAD_8BIT):
    if 'llama' in model_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if lora_path:

        # fix the path for local checkpoint
        lora_bin_path = os.path.join(lora_path, "adapter_model.bin")
        print(lora_bin_path)
        if not os.path.exists(lora_bin_path) and args.use_local:
            pytorch_bin_path = os.path.join(lora_path, "pytorch_model.bin")
            print(pytorch_bin_path)
            if os.path.exists(pytorch_bin_path):
                os.rename(pytorch_bin_path, lora_bin_path)
                warnings.warn(
                    "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
                )
            else:
                assert ('Checkpoint is not Found!')

        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=LOAD_8BIT,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
            model = SteamGenerationMixin.from_pretrained(
                model, lora_path, torch_dtype=torch.float16, device_map={"": 0}
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = SteamGenerationMixin.from_pretrained(
                model,
                lora_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = SteamGenerationMixin.from_pretrained(
                model,
                lora_path,
                device_map={"": device},
            )
    else:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=LOAD_8BIT,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map={"": device}, low_cpu_mem_usage=True
            )

    if not LOAD_8BIT:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, tokenizer


print(f'loading model: {model_name_list}')
model_tokenizer_list = [
    load_model_and_tokenizer(model_path, lora_path, LOAD_8BIT)
    for model_path, lora_path, LOAD_8BIT in zip(model_path_list, lora_path_list, load_8bit_list)
]


def evaluate(
    inputs,
    history,
    model_name,
    temperature=0.3,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=128,
    min_new_tokens=1,
    max_memory=1024,
    do_sample=True,
    prompt_type='Conversation',
    **kwargs,
):
    def generate_prompt_and_tokenize0(data_point, maxlen):
        # cutoff the history to avoid exceeding length limit
        init_prompt = PROMPT_DICT['prompt']
        init_ids = tokenizer(init_prompt)['input_ids']
        seqlen = len(init_ids)
        input_prompt = PROMPT_DICT['input'].format_map(data_point)
        input_ids = tokenizer(input_prompt)['input_ids']
        seqlen += len(input_ids)
        if seqlen > maxlen:
            raise Exception('>>> The input question is too long! Cosidering increase the Max Memory value or decrease the length of input! ')
        history_prompt = ''
        for history in data_point['history']:
            history_prompt+= PROMPT_DICT['history'].format_map(history) 
        # cutoff
        history_ids = tokenizer(history_prompt)['input_ids'][-(maxlen - seqlen):]
        input_ids = init_ids + history_ids + input_ids
        return input_ids

    def postprocess0(text, render=True):
        # clip user
        text = text.split("### Assistant:")[1].strip()
        text = text.replace('�','').replace("Belle", "Vicuna")
        return text

    def generate_prompt_and_tokenize1(data_point, maxlen):
        input_prompt = "\n".join(["User:" + i['input']+"\n"+"Assistant:" + i['output'] for i in data_point['history']]) + "\nUser:" + data_point['input'] + "\nAssistant:"
        input_prompt = input_prompt[-maxlen:]
        input_prompt = PROMPT_DICT['prompt'].format_map({'input':input_prompt})
        input_ids = tokenizer(input_prompt)["input_ids"]
        return input_ids

    def postprocess1(text, render=True):
        output = text.split("### Response:")[1].strip()
        output = output.replace("Belle", "Vicuna")
        printf('>>> output:', output)
        if '###' in output:
            output = output.split("###")[0]
        if 'User' in output:
            output = output.split("User")[0]
        output = output.replace('�','') 
        if render:
            # fix gradio chatbot markdown code render bug
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "```" in line:
                    if line != "```":
                        lines[i] = f'<pre><code class="language-{lines[i][3:]}">'
                    else:
                        lines[i] = '</code></pre>'
                else:
                    if i > 0:
                        lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;").replace("__", '\_\_')
            output =  "".join(lines)
            # output = output.replace('<br/><pre>','\n<pre>') work for html; but not for gradio
        return output
    
    def generate_prompt_and_tokenize2(data_point, maxlen):
        input_prompt = data_point['input']
        input_prompt = input_prompt[-maxlen:]
        input_prompt = PROMPT_DICT['prompt'].format_map({'input':input_prompt})
        input_ids = tokenizer(input_prompt)["input_ids"]
        return input_ids
    
    def postprocess2(text):
        # output = text.split("### Response:")[1].strip().split('###')[0]
        output = text.strip().split('###')[0]
        return output


    print(f'You are choosing model: {model_name}')
    model, tokenizer = model_tokenizer_list[model_name_dict[model_name]]

    PROMPT_DICT0 = {
        'prompt': (
            "The following is a conversation between an AI assistant called Assistant and a human user called User."
            "Assistant is is intelligent, knowledgeable, wise and polite.\n\n"
        ),
        'history': (
            "User:{input}\n\nAssistant:{output}\n\n"
        ),
        'input': (
            "User:{input}\n\n### Assistant:"
        ),
        'preprocess': generate_prompt_and_tokenize0,
        'postprocess': postprocess0,
    }
    PROMPT_DICT1 = {
        'prompt': (
            "The following is a conversation between an AI assistant called Assistant and a human user called User.\n\n"
            "### Instruction:\n{input}\n\n### Response:"
        ),
        'preprocess': generate_prompt_and_tokenize1,
        'postprocess': postprocess1,
    }
    PROMPT_DICT2 = {
        'prompt': (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{input}\n\n### Response:"
        ),
        'preprocess': generate_prompt_and_tokenize2,
        'postprocess': postprocess2,
    }

    PROMPT_DICT = None
    if 'vicuna' in model_name:
        PROMPT_DICT = PROMPT_DICT2
    elif 'lora' in model_name:
        PROMPT_DICT = PROMPT_DICT2
    elif prompt_type == 'Conversation':
        PROMPT_DICT = PROMPT_DICT0
    elif prompt_type == 'Instruction':
        PROMPT_DICT = PROMPT_DICT1
    else:
        raise Exception('not support')
    
    history = [] if history is None else history
    data_point = {
        'history': history,
        'input': inputs,
    }
    printf(data_point)
    input_ids = PROMPT_DICT['preprocess'](data_point, max_memory)
    printf('>>> input prompts:', tokenizer.decode(input_ids))
    input_ids = torch.tensor([input_ids]).to(device) # batch=1
    printf(input_ids.shape)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        do_sample=do_sample,
        **kwargs,
    )
    
    return_text = [(item['input'], item['output']) for item in history]
    out_memory =False
    with torch.no_grad():
        # 流式输出 / 打字机效果
        # streamly output / typewriter style
        if args.use_typewriter:
            try:
                streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True)
                generation_kwargs = dict(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    streamer=streamer,
                )
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                generation_output = ""
                for new_text in streamer:
                    generation_output += new_text
                    show_text = PROMPT_DICT['postprocess'](generation_output)+" ▌"
                    printf(show_text)
                    yield return_text +[(inputs, show_text)], history

            except torch.cuda.OutOfMemoryError:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                out_memory=True
            streamer.end()
            return_len = len(show_text)
            if return_len > 1:
                show_text = show_text[:-1]
            if out_memory==True:
                out_memory=False
                show_text+= '<p style="color:#FF0000"> [GPU Out Of Memory] </p> '
            output = show_text
            history.append({
                'input': inputs,
                'output': output,
            })
            printf(show_text)
            return_text += [(inputs, show_text)]
            yield return_text, history
        else:
            try:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                s = generation_output.sequences[0]
                output = tokenizer.decode(s)
                output = PROMPT_DICT['postprocess'](output)
                history.append({
                    'input': inputs,
                    'output': output,
                })
                return_text += [(inputs, output)]
                yield return_text, history
            except torch.cuda.OutOfMemoryError:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                show_text = '<p style="color:#FF0000"> [GPU Out Of Memory] </p> '
                printf(show_text)
                return_text += [(inputs, show_text)]
                yield return_text, history



# gr.Interface对chatbot的clear有bug，因此我们重新实现了一个基于gr.block的UI逻辑
# gr.Interface has bugs to clear chatbot's history,so we customly implement it based on gr.block
with gr.Blocks() as demo:
    fn = evaluate
    title = gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>"
        + f"E2E AIOK optimized LLM"
        + "</h1>"
    )
    description = gr.Markdown(
        f"For demo purpose only. This demo is intended to show the draft UI of LLM on Ray workflows in Mt.Whitney, demonstrating the functionality readiness of LLM on Ray workflow, with Multiple LLM supported. "
        "Currently, we support GPT-2/GPT-J/LLaMA series models. We are planning to support more GPT-like models."
    )
    history = gr.components.State()
    with gr.Row().style(equal_height=False):
        with gr.Column(variant="panel"):
            input_component_column = gr.Column()
            with input_component_column:
                input = gr.components.Textbox(
                    lines=2, label="Input", placeholder="Please input your question."
                )
                model_name = gr.Dropdown(choices=model_name_list, label='model list', value=model_name_list[0])
                with gr.Row():
                    cancel_btn = gr.Button('Cancel')
                    submit_btn = gr.Button("Submit", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop", visible=False)
                temperature = gr.components.Slider(minimum=0, maximum=1, value=0.3, label="Temperature")
                topp = gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p")
                topk = gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k")
                beam_number = gr.components.Slider(minimum=1, maximum=10, step=1, value=1, label="Beams Number")
                max_new_token = gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=128, label="Max New Tokens"
                )
                min_new_token = gr.components.Slider(
                    minimum=1, maximum=100, step=1, value=1, label="Min New Tokens"
                )
                max_memory = gr.components.Slider(
                    minimum=0, maximum=2048, step=1, value=256, label="Max Memory"
                )
                do_sample = gr.components.Checkbox(label="do sample", value=True)
                # must be str, not number !
                type_of_prompt = gr.components.Dropdown(
                    ['Conversation', 'Instruction'], value='Instruction', label="Prompt Type", info="select the specific prompt; use after clear history"
                )
                input_components = [
                    input, history, model_name, temperature, topp, topk, beam_number, max_new_token, min_new_token, max_memory, do_sample, type_of_prompt
                ]
                input_components_except_states = [input, model_name, temperature, topp, topk, beam_number, max_new_token, min_new_token, max_memory, do_sample, type_of_prompt]
            with gr.Row():
                reset_btn = gr.Button("Reset Parameter")
                clear_history = gr.Button("Clear History")


        with gr.Column(variant="panel"):
            chatbot = gr.Chatbot().style(height=1024)
            output_components = [ chatbot, history ]  

        def wrapper(*args):
            # here to support the change between the stop and submit button
            try:
                for output in fn(*args):
                    output = [o for o in output]
                    # output for output_components, the rest for [button, button]
                    yield output + [
                        gr.Button.update(visible=False),
                        gr.Button.update(visible=True),
                    ]
            finally:
                yield [{'__type__': 'generic_update'}, {'__type__': 'generic_update'}] + [ gr.Button.update(visible=True), gr.Button.update(visible=False)]

        def cancel(history, chatbot):
            if history == []:
                return (None, None)
            return history[:-1], chatbot[:-1]

        extra_output = [submit_btn, stop_btn]

        pred = submit_btn.click(
            wrapper, 
            input_components, 
            output_components + extra_output, 
            api_name="predict",
            scroll_to_output=True,
            preprocess=True,
            postprocess=True,
            batch=False,
            max_batch_size=4,
        )
        submit_btn.click(
            lambda: (
                submit_btn.update(visible=False),
                stop_btn.update(visible=True),
            ),
            inputs=None,
            outputs=[submit_btn, stop_btn],
            queue=False,
        )
        stop_btn.click(
            lambda: (
                submit_btn.update(visible=True),
                stop_btn.update(visible=False),
            ),
            inputs=None,
            outputs=[submit_btn, stop_btn],
            cancels=[pred],
            queue=False,
        )
        cancel_btn.click(
            cancel,
            inputs=[history, chatbot],
            outputs=[history, chatbot]
        )
        reset_btn.click(
            None, 
            [],
            (
                # input_components ; don't work for history...
                input_components_except_states
                + [input_component_column]
            ),  # type: ignore
            _js=f"""() => {json.dumps([
                getattr(component, "cleared_value", None) for component in input_components_except_states ] 
                + ([gr.Column.update(visible=True)])
                + ([])
            )}
            """,
        )
        clear_history.click(lambda: (None, None), None, [history, chatbot], queue=False)

demo.queue().launch(share=args.share_link!=0, inbrowser=True, server_name='0.0.0.0')
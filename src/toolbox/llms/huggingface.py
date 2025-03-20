import torch.nn as nn

import transformers

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


class LangChainEmbedding(nn.Module):
    def __init__(self, model_name, device):
        super(LangChainEmbedding, self).__init__()
        self.device = device
        self.model_name = model_name
        
        self.embedder = HuggingFaceEmbeddings(model_name = model_name)
    
    
    def forward(self, input):
        return self.embedder.embed_query(input)


class LangChainToken2Token(nn.Module):
    def __init__(self, model_name, device, batch_size = 1, 
                 model_kwargs = {}, pipeline_kwargs = {}, token_kwargs = {}, 
                 prompt_template = None):
        super(LangChainToken2Token, self).__init__()
        self.device = device
        self.model_name = model_name
        
        # Do not append the prompt into the output.
        pipeline_kwargs['return_full_text'] = False
        token_kwargs['padding_side'] = 'left'
        
        # self.model = HuggingFacePipeline.from_model_id(
        #     model_id = model_name,
        #     task = 'text-generation',
        #     batch_size = batch_size,
        #     model_kwargs = model_kwargs,
        #     pipeline_kwargs = pipeline_kwargs,
        #     device_map = device
        # )
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs, device_map = self.device)
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **token_kwargs)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        pipe = transformers.pipeline("text-generation", model = model, tokenizer = tokenizer, \
                                    batch_size = batch_size, **pipeline_kwargs)
        self.model = HuggingFacePipeline(pipeline = pipe)
        
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.model


    def forward(self, input):
        answers = self.chain.batch(input)
        return answers


if __name__ == '__main__':
    # model_name = '/home/undesired/coderepo/workflow/llms'
    # model_kwargs = {'gguf_file': 'Llama-3.2-3B-Instruct-IQ4_XS.gguf'}
    model_name = 'ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3'
    
    template = 'Please answer the following question: {question}.'
    model = LangChainToken2Token(model_name, batch_size = 4, \
                                 device = 'cuda:0', prompt_template = template, \
                                 model_kwargs = {'temperature': 0.8},
                                 pipeline_kwargs = {'max_new_tokens': 50}, 
                                 token_kwargs = {'model_max_length': 16384, 'truncation': True})
    
    questions = []
    for i in range(10):
        questions.append({'question': f'What is the number {i} in French?'})
        questions.append({'question': f'What is the number {i} in English? Please answer it by a single word.'})

    answers = model.forward(questions, )
    
    embedder = LangChainEmbedding(model_name = 'mixedbread-ai/mxbai-embed-large-v1', device = 'cuda:0')
    results = embedder(answers[0])
    
    for answer in answers:
        print(answer)
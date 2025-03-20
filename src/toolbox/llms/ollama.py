import requests

from torch import nn as nn

from src.toolbox.llms.utils import slash_join


data_dict_based_on_task = {
    'generate': [lambda x, model_name, ollama_options: \
            {
                "model": model_name,
                "prompt": x,
                "stream": False,
                "options": ollama_options
            }, 'response'],
    'embed': [lambda x, model_name, ollama_options: \
            {
                "model": model_name,
                "input": x
            }, 'embeddings']
}


class OllamaToken2Token(nn.Module):
    def __init__(self, ollama_url, device):
        super(OllamaToken2Token, self).__init__()
        self.device = device
        self.ollama_url = ollama_url


    def forward(self, prompt, model_name, task = 'generate', ollama_options = {"num_ctx": 8192, "num_predict": 5000}):
        ollama_url = slash_join(self.ollama_url, 'api/', task + '/')
        template, response_key = data_dict_based_on_task[task]
        sent_json = template(prompt, model_name, ollama_options)
        
        response = requests.post(ollama_url, headers = {}, json = sent_json)
        return response.json()[response_key]
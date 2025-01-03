import transformers

lm_module_location = {
    'bert': transformers.models.bert.modeling_bert.BertModel,
    'gpt2': transformers.models.gpt2.modeling_gpt2.GPT2Model,
    'llama': transformers.models.llama.modeling_llama.LlamaModel,
    'qwen': transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
}
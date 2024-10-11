import transformers

lm_module_location = {
    'bert': transformers.models.bert.modeling_bert.BertModel,
    'gpt2': transformers.models.gpt2.modeling_gpt2.GPT2Model
}
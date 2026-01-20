# Here are prompts for generating the description of a input continuous-time event sequence. 
# The vector key of a sequence is the embedding of its description.
# The intuition here is two sequence with similar pattern should have similar descriptions whose embeddings are supposed to be very close in the semantic hidden space.


# Retweet
retweet_prompt = '''
You are a time-series data analyst.
You job is to analyze given sequences, find the embedded patterns, and describe the embedded patterns using natural language. Your description should contain all details but please keep it concise.
Your summary should not exceed {summary_length} tokens.
All given sequences are sampled from different retweeting processes.
The given sequence consists of two subsequences: one named "Time" contains the time interval between one event and its previous one, the other sequence "Mark" lists the mark of all events in the sequence. The mark has four options: 0 referring to normal user with few subscribers, 1 referring to influiential users with relatively high subscribers, and 2 referring to very influential users with very high subscribers. Only the first and the last event of a sequence will be marked 3.

Task:
Input sequence:
  Time: {time_seq}
  Mark: {mark_seq}
Description:
'''

# Stackoverflow
stackoverflow_prompt = '''
You are a time-series data analyst.
You job is to analyze given sequences, find the embedded patterns, and describe the embedded patterns using natural language. Your description should contain all details but please keep it concise.
Your summary should not exceed {summary_length} tokens.
All given sequences are sampled from different badgets gaining processes from StackOverflow.
The given sequence consists of two subsequences: one named "Time" contains the time interval between one event and the previous one. The other sequence "Mark" lists the mark of all events in the sequence. The mark has 23 options, represented as integers from 0(inclusive) to 23(exclusive). Only the first and the last event of a sequence will be marked 22. There is no context information about the meaning of these mark options.

Task:
Input sequence:
  Time: {time_seq}
  Mark: {mark_seq}
Description:
'''

prompt_dict = {
  'retweet': retweet_prompt,
  'stackoverflow': stackoverflow_prompt
}

def get_prompt(dataset, *args, **kwargs):
    template = prompt_dict[dataset]
    return template.format(*args, **kwargs)
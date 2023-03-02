_B='./chatgpt_model'
_A='google/flan-t5-xxl'
import os,torch
os.environ['TRANSFORMERS_CACHE']='./'
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
model_path='./'
model=AutoModelForSeq2SeqLM.from_pretrained(_A)
tokenizer=AutoTokenizer.from_pretrained(_A)
device='cuda'if torch.cuda.is_available()else'cpu'
model.to(device)
input_instruction=input('Enter input instruction: ')
state_count=input('Enter state count (default is 64): ')
if state_count=='':state_count=64
else:state_count=int(state_count)
past_key_values=None
last_past_key_values=None
state_counter=0
while True:
	if state_counter==0:user_input=input(input_instruction+'\nUser: ')
	else:user_input=input('User: ')
	if user_input.lower()=='exit':break
	try:
		input_ids=tokenizer.encode(input_instruction+'\n'+user_input,return_tensors='pt').to(device);output_ids=model.generate(input_ids=input_ids,past_key_values=past_key_values,max_length=100000,length_penalty=1.0,temperature=0.8,do_sample=True);past_key_values=model.get_encoder()(input_ids).past_key_values;state_counter+=1
		if state_counter>=state_count:past_key_values=None;state_counter=0;print('Model state has been reset.')
	except Exception as e:print('Error in processing:',e);continue
	try:output_text=tokenizer.decode(output_ids[0],skip_special_tokens=True);print('Flan-T5:',output_text)
	except Exception as e:print('Error in decoding:',e)

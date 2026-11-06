from tokenization import tokenizer

filepath = "C:/Users/sd809918/Downloads/Project_code/WhatsApp_Chat/_chat.txt"

data, encode, decode, vocab_size = tokenizer(filepath)

n = 0.9*len(data)
train_data = data[:n]
val_data = data[n:]
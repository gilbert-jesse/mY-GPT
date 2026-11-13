from tokenization import tokenizer
import torch
from bigram_model import BiagramLanguageModel


filepath = "C:/Users/sd809918/Downloads/Project_code/WhatsApp_Chat/_chat.txt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_iter = 3000
eval_interval = 300
learning_rate = 1e-2


data, encode, decode, vocab_size = tokenizer(filepath)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for prediction?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix  = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



model = BiagramLanguageModel(vocab_size)
model = model.to(device)

# create a PyTorch optimizer
optimizer  = torch.optim.AdamW(model.parameters(), lr=learning_rate)

batch_size = 32
for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
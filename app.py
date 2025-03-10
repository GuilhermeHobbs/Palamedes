from flask import Flask, request
from flask_cors import CORS

import numpy as np
import torch
import torch.nn as nn
import sys
import math

from torch.nn import functional as F

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

block_size = 100
vocab_size = 34
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embd = 400 # 64
n_head = 16  # 4
n_layer = 12 # 8
dropout = 0.2
# ------------

@torch.no_grad()
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


model = BigramLanguageModel()
m = model.to(device)

m.load_state_dict(torch.load('winMINI.pth', map_location=torch.device('cpu')))

model.eval()  # Disable dropout

context = torch.Tensor([[0]]).int().to(device)  

red_o = {0,1,2,3,4,5,6,7,8,9,10,11,12}
white_o = {21,22,23,24,25,26,27,28,29,30,31,32}


@app.route('/move')
def ask_name():
    global m,context,white_o,red_o
     
    resp = request.args.get('resp', '')  
    
    if resp=='':
        context = torch.Tensor([[0]]).int().to(device)
        red_o = {0,1,2,3,4,5,6,7,8,9,10,11,12}
        white_o = {21,22,23,24,25,26,27,28,29,30,31,32}
        return ""

    resp = torch.tensor([[int(x) for x in resp.split('-')]])
    context = torch.cat([context, resp], dim=1)

    for i in range(context.shape[1]):
      if context[0, i].item()==33:  
        if (context[0, i-1].item()+context[0, i+1].item())%16 < 8:
             middle = math.floor((context[0, i-1].item()+context[0, i+1].item())/2)
        else:
             middle = math.ceil((context[0, i-1].item()+context[0, i+1].item())/2)
        
        white_o.remove(middle)
        red_o.add(context[0, i+1].item())
    else:
       red_o.add(b) 
    
    k=0
    i=0
    quantos_33 = 0
    
    while i-quantos_33<2 and not(i==5 and quantos_33==2):

      logits, _ = m(context.int())
      logits = logits[-1,-1] 
      _, b = torch.topk(logits, k+1)
      print(torch.topk(logits, k+1))
      b = b[k]
      print(i,"antes: ",b,context,white_o,red_o, flush=True)  # Force immediate flushing
      sys.stdout.flush()

      if b==33:
       d = torch.cat([context, torch.Tensor([[33]]).to(device)], dim=1)
       l, _ = m(d.int())
       l = l[-1,-1] 
       b = l.argmax()

       if (context[0,-1].item()+b.item())%16 < 8:
        middle = math.floor((context[0,-1].item()+b.item())/2)
       else:
         middle = math.ceil((context[0,-1].item()+b.item())/2)
       # False False True False True 11 True False 16.0

       print( b.item()<context[0,-1].item()-7,b.item(),context[0,-1].item(), flush=True)  # Force immediate flushing
       sys.stdout.flush()   
       if (not (b.item() in white_o.union(red_o) or b.item()>context[0,-1].item()+9 or b.item()<context[0,-1].item()-9)) and middle in white_o:
          print("else:",b.item(),white_o.union(red_o),context[0,-1].item(),"comer",middle, flush=True)  # Force immediate flushing
          sys.stdout.flush()
          if (context[0,-1].item() in white_o):
            red_o.remove(middle)
            white_o.remove(context[0,-1].item())
            white_o.add(b.item())
          if (context[0,-1].item() in red_o):
            white_o.remove(middle)
            red_o.remove(context[0,-1].item())
            red_o.add(b.item())
          print (i,"TWO",context[0,-2],context[0,-1], flush=True)  # Force immediate flushing
          sys.stdout.flush()
          if i%2 == 1:
            i+=1
          else:
            i+=2
          context = torch.cat([context, torch.Tensor([[33,b.item()]]).to(device)], dim=1)
          print (i,"TWO depois",context[0,-2],context[0,-1], flush=True)  # Force immediate flushing
          sys.stdout.flush()
          quantos_33 += 1
          continue
       print("NOT else:",b.item(),white_o.union(red_o),context,middle, flush=True)  # Force immediate flushing
       sys.stdout.flush()
      
      if i%2 == 1:
       print("i%2 == 1")
       if b.item() in white_o.union(red_o) or abs(b.item() - context[0,-1].item())<3 or abs(b.item() - context[0,-1].item())>5:   #b.item()>c[0,-1].item()+9 or b.item()<c[0,-1].item()-9:
        if k>3 and context[0,-2].item() != 33:
          context = context[:, :-1]
          i-=1
          k = 1
          continue
        else:
          k+=1
          continue
       else:
        if (context[0,-1].item() in white_o):
            white_o.remove(context[0,-1].item())
            white_o.add(b.item())
        if (context[0,-1].item() in red_o):
            red_o.remove(context[0,-1].item())
            red_o.add(b.item())

      if i%2 == 0 and quantos_33 > 0 and b.item() != 33:
          i+=1
          print("BREAK",i,b)
          break  
        
      if i%2 == 0 and b.item() not in white_o.union(red_o):
        print("i%2 == 0 and i>0 and b.item() not in white_o.union(red_o)", b.item(), flush=True)  # Force immediate flushing
        sys.stdout.flush()
        k+=1
        continue
  
      context = torch.cat([context, torch.Tensor([[b.item()]]).to(device)], dim=1)
      print ("depois: ",b,context,white_o,red_o,i,k,quantos_33, flush=True)  # Force immediate flushing
      sys.stdout.flush()
      k=0
      i+=1

    
    print("Cooontext:",context, flush=True)  # Force immediate flushing
    sys.stdout.flush()
    
    if i==2:
        return str(context[0,-2].item())+"-"+str(context[0,-1].item())  
    if i==3:
        return str(context[0,-3].item())+"-33-"+str(context[0,-1].item())
    if i==5:
        return str(context[0,-5].item())+"-33-"+str(context[0,-3].item())+"-33-"+str(context[0,-1].item()) 
    else:
        print("ERROR:",i,context)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render requires explicit host/port

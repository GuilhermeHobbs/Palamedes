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

block_size = 120
vocab_size = 34
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embd = 400 # 64
n_head = 16  # 4
n_layer = 24
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

m.load_state_dict(torch.load('last1704.pth', map_location=torch.device('cpu')))

model.eval()  # Disable dropout

context = torch.Tensor([[0]]).int().to(device)  

white_o = {1,2,3,4,5,6,7,8,9,10,11,12}
red_o = {21,22,23,24,25,26,27,28,29,30,31,32,0}


@app.route('/move')
def ask_name():
    global m,context,white_o,red_o
     
    resp = request.args.get('move', '')  
    
    if resp=='':
        context = torch.Tensor([[0]]).int().to(device)
        white_o = {1,2,3,4,5,6,7,8,9,10,11,12}
        red_o = {21,22,23,24,25,26,27,28,29,30,31,32,0}
        print("RESTART", flush=True)  # Force immediate flushing
        sys.stdout.flush()
        return ""

    resp = torch.tensor([[int(x) for x in resp.split('-')]])
    context = torch.cat([context, resp], dim=1)

    context = context[:, -block_size+6:]
    
    print("Resp", context, flush=True)  # Force immediate flushing
    sys.stdout.flush()

    red_o.remove(resp[0, 0].item())
    if resp[0, 1].item() != 33:
        red_o.add(resp[0, 1].item())
    
    else:
        i=1
        while i<resp.shape[1] and resp[0, i].item() == 33:
          if (resp[0, i-1].item()+resp[0, i+1].item())%16 < 8:
             middle = math.floor((resp[0, i-1].item()+resp[0, i+1].item())/2)
          else:
             middle = math.ceil((resp[0, i-1].item()+resp[0, i+1].item())/2)
          white_o.remove(middle)
          i+=2
        red_o.add(resp[0, i-1].item())
    print(red_o,flush=True)  # Force immediate flushing
    sys.stdout.flush()     
    print(white_o,flush=True)  # Force immediate flushing
    sys.stdout.flush()

    
     
    i=0    
    a=0
    k=0 
    End_of_jump = False
    while End_of_jump == False:
     print("Recomeçou",k) 
     while a not in white_o:  
      logits, _ = m(context.int())
      logits = logits[-1,-1] 
      _, a = torch.topk(logits, k+1)
      print(torch.topk(logits, k+1), flush=True)  # Force immediate flushing
      sys.stdout.flush()
      a = a[k].item()
      print(k,"a[k].item()",a, flush=True)  # Force immediate flushing
      sys.stdout.flush()
      k += 1  
     
     context = torch.cat([context, torch.Tensor([[a]]).to(device)], dim=1)
     print(i,"Context first",context, flush=True)  # Force immediate flushing
     sys.stdout.flush()
     j=0
     b=0
     while b in red_o.union(white_o) and j<4:          
        logits, _ = m(context.int())
        logits = logits[-1,-1] 
        _, b = torch.topk(logits, j+1)
        print(torch.topk(logits, j+1), flush=True)  # Force immediate flushing
        sys.stdout.flush()
        b = b[j].item()
        print(j,"b[k].item()",b, flush=True)  # Force immediate flushing
        sys.stdout.flush() 
        j+=1    
       
     if j==4:
         context = context[:, :-1]
         k = 1    
         a = 0 
         continue
     
     context = torch.cat([context, torch.Tensor([[b]]).to(device)], dim=1) 
     i=2
        
     print("Context after",context, flush=True)  # Force immediate flushing
     sys.stdout.flush()   
     if b==33:
        logits, _ = m(context.int())
        logits = logits[-1,-1] 
        c = logits.argmax().item()

        if (a+c)%16 < 8:
            middle = math.floor((a+c)/2)
        else:
            middle = math.ceil((a+c)/2)     
        print("Middle",middle, flush=True)  # Force immediate flushing
        sys.stdout.flush()    
        if c in red_o.union(white_o) or middle not in red_o:
           context = context[:, :-2]
           print("if c in red_o.union(white_o) or middle not in red_o:",c) 
           k+=1
           a=0 
           continue 
        else:
           context = torch.cat([context, torch.Tensor([[c]]).to(device)], dim=1)
           k=0
           print("Context [[c]]",context, flush=True)  # Force immediate flushing
           sys.stdout.flush()    
           white_o.remove(a) 
           print("white_o.remove(a)", a, flush=True)  # Force immediate flushing
           sys.stdout.flush()
           white_o.add(c)
           print("white_o.add(c)", c, flush=True)  # Force immediate flushing
           sys.stdout.flush() 
           red_o.remove(middle)
           print("red_o.remove(middle)", middle, flush=True)  # Force immediate flushing
           sys.stdout.flush()  
           old_b = c   
           i=3 
           while End_of_jump == False:
            logits, _ = m(context.int())
            logits = logits[-1,-1] 
            a = logits.argmax().item()
            print("a = logits.argmax().item()",a, flush=True)  # Force immediate flushing
            sys.stdout.flush()    
            if a != 33:
                End_of_jump = True
            else:
                print("if b != 33",old_b, b, flush=True)  # Force immediate flushing
                sys.stdout.flush()        
                
                if b != 33:
                    old_b = b
                context = torch.cat([context, torch.Tensor([[33]]).to(device)], dim=1)      
                logits, _ = m(context.int())
                logits = logits[-1,-1] 
                b = logits.argmax().item()
                print("b = logits.argmax().item()",b, flush=True)  # Force immediate flushing
                sys.stdout.flush()        
                if (old_b+b)%16 < 8:
                    middle = math.floor((old_b+b)/2)
                else:
                    middle = math.ceil((old_b+b)/2)     
                
                if b in red_o.union(white_o) or b==33 or middle not in red_o:   
                   context = context[:, :-1]
                   print("if b in red_o.union(white_o) or b==33 or middle not in red_o:",context)   
                   End_of_jump = True
                else:
                  context = torch.cat([context, torch.Tensor([[b]]).to(device)], dim=1)  ###########    
                  print(context)
                  red_o.remove(middle)
                  print("red_o.remove(middle)", middle, flush=True)  # Force immediate flushing
                  sys.stdout.flush()  
                  print("white_o.remove(old_b)", old_b, flush=True)  # Force immediate flushing
                  sys.stdout.flush()    
                  white_o.remove(old_b)
                  white_o.add(b)
                  print("white_o.add(b)", b, flush=True)  # Force immediate flushing
                  sys.stdout.flush()      
                  i+=2
                  print("i+=2",i)  
     else:  
         white_o.remove(a)
         print("white_o.remove(a)", a, flush=True)  # Force immediate flushing
         sys.stdout.flush()      
         white_o.add(b)
         print("white_o.add(b)", b, flush=True)  # Force immediate flushing
         sys.stdout.flush()      
         break
    print(i,"CONTEXT:",context, flush=True)  # Force immediate flushing
    sys.stdout.flush()           
    
    moves = context[0, -i:].int().tolist()
    
    # Convert list of numbers to a hyphen-separated string
    return "-".join(map(str, moves))
    
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render requires explicit host/port

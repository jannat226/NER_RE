import torch 
data = torch.load('ner_labels.pt')
for i in range(10):
    print(data[i])

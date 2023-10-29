import torch
a = torch.zeros(16,48)
b = torch.ones(16,48)
c = torch.ones(48)*0.5
label_a = torch.zeros(16)
label_b = torch.ones(16)
data = torch.cat([a,b])
label = torch.cat([label_a,label_b])
# print((data-c))
# loss = (((label-1)**2)*(data-c)) + ((label)*(data-c)) 
# print(loss.shape)
# print((((label-1)**2)*(data-c)))
mse = torch.mean(((data-c)[label==1]**2),dim=1)
semi = 1/torch.mean(((data-c)[label==0]**2),dim=1)
print(semi)
print(mse)

loss = torch.mean(torch.cat([mse,semi]))
print(loss)
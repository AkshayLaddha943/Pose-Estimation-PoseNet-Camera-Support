# function to store model checkpoint
import torch 

LEARNING_RATE = 0.0001
EXPERIMENTS_PATH="C:/Object-Detection/PoseEstimation/output"

def checkpoint_model(model, optimizer, epoch, batch):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, EXPERIMENTS_PATH + "posenet_" + str(epoch) + "_" + str(batch) + '.pth')
    

def optim(model):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def schedule(optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [8], 0.1)
    return scheduler
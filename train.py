import sys
sys.path.append("C:/Object-Detection/")

from PoseEstimation import conv, data, utils, metrics
NUM_EPOCHS = 10

#model prep
model = conv.EncoderDecoderNet()
model.cuda()
model.train()

def main():
    for k in range(NUM_EPOCHS):
        for i_batch, j_batch in enumerate(data.train_loader):
            
            input_img = j_batch['input_img'].to('cuda')
            heatmap = j_batch['heatmap'].to('cuda')
            validity = j_batch['validity'].to('cuda')
            
            #clearning out old grad
            optimizer = utils.optim(model)
            optimizer.zero_grad()
            
            output = model(input_img)
            loss = metrics.loss_fn(output, heatmap, validity)
            
            #compute grad
            loss.backward()
            
            #update param
            optimizer.step()
            
            if i_batch % 100 == 0:
                
                acc_train = metrics.accuracy(output.detach().cpu().numpy(),heatmap.detach().cpu().numpy())
                acc_val = metrics.get_accuracy(model,data.val_loader)
                print("Epoch", k, "Batch", i_batch, "Training Loss: ", loss.item(), "Train Accuracy: ", acc_train, "Val Accuracy: ", acc_val)
            
            if i_batch % 500 == 0:        
                utils.checkpoint_model(model, optimizer, k, i_batch)
        
        utils.schedule.step(optimizer)
    utils.checkpoint_model(model, optimizer, k, i_batch)

if __name__ == "__main__":
        main()
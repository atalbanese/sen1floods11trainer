import torch
import torch.utils.data
from pathlib import Path
import random
import csv
import torchvision.models as models
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import flooddata as fl
import losses
import tests as t
from torch.utils.tensorboard import SummaryWriter
import nets


## DATA LOADING CLASSES/FUNCTIONS

# Get data splits for training, validation, and testing. Made using get_distributions.py to have equal distributions of
# number of water pixels in each split
def get_data_splits(data_type, data_dir):
    train_path = data_dir.joinpath(Path("./splits/" + data_type + "/train_split.csv"))
    valid_path = data_dir.joinpath(Path("./splits/" + data_type + "/valid_split.csv"))
    test_path = data_dir.joinpath(Path("./splits/" + data_type + "/test_split.csv"))
    path_list = [train_path, valid_path, test_path]
    out_list = []
    for path in path_list:
        file_list = []
        with open(path, newline='') as csvfile:
            fileread = csv.reader(csvfile, delimiter=',')
            for row in fileread:
                to_append = data_dir.joinpath(Path(row[0])), data_dir.joinpath(Path(row[1]))
                file_list.append(to_append)
        out_list.append(file_list)
    return out_list


## TRAINING AND TESTING LOOPS

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss, train_iou, train_acc = 0.0, 0.0, 0.0
    model.train()
    optimizer.zero_grad()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch, (inputs, labels, item) in enumerate(dataloader, 0):
        # Copy to cuda if available

        inputs, labels = inputs.to(device), labels.to(device)

        # forward
        # outputs = model(inputs)['out'].cuda()
        outputs = model(inputs)
        labels = labels.long()
        loss = loss_fn(outputs, labels)
        train_loss += loss
        train_iou += t.computeIOU(outputs, labels)
        train_acc += t.computeAccuracy(outputs, labels)

        # backward + optimize
        # zero the parameter gradients
        # Accumulates multiple smaller batches if batch size is greater than 8. This prevents memory allocation errors.
        loss.backward()
        if len(inputs) > 8:
            if (batch + 1) % int(len(inputs) / 8) == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()

        # print statistics
        if batch % 16 == 0:
            loss, current = loss.item(), batch * len(inputs)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
    print(f"Train Avg loss: {train_loss / size:>8f}, Train_IOU: {train_iou/size} \n")
    return {"Train_Loss": train_loss / size, "Train_IOU": train_iou / size, "Train_Acc": train_acc / size}


# Run all images in the validation dataloader through the network, computing mean IoU and accuracy as outputs
def valid_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    valid_loss, valid_iou, valid_acc = 0.0, 0.0, 0.0
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for images, labels, item in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            # pred = model(images)['out'].cuda()
            pred = model(images)
            valid_loss += loss_fn(pred, labels).item()
            valid_iou += t.computeIOU(pred, labels)
            valid_acc += t.computeAccuracy(pred, labels)
    print(f"Validation Avg loss: {valid_loss / size:>8f} \n")
    return {"Valid_Loss": valid_loss / size, "Valid_IOU": valid_iou / size, "Valid_Acc": valid_acc / size}


#### RUN
# Load training, validation, and testing datasets.
# TODO: testing is performed in test_model.py, it can be removed here
def load_datasets(dataset, data_dir):
    train, valid, test = get_data_splits(dataset, data_dir)

    train_data = fl.FloodTrainData(train)
    valid_data = fl.FloodValidData(valid)
    test_data = fl.FloodValidData(test)
    return train_data, valid_data, test_data


def train_sen(config=None, checkpoint_dir=None, data_dir=None):
    LR = config["lr"]
    #LR = .00635
    EPOCHS = config["epochs"]
    RUNNAME = "Sen1Floods11"
    DATASET = config["dataset"]
    RANDOM_SEED = random.randrange(100000)
    MODEL_NAME = config["model_name"]

    # Get datasets
    train_data, valid_data, test_data = load_datasets(DATASET, data_dir)
    # Create data loaders from datasets
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config["bs"],
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=4,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4)
    # ResNet specific model prep
    # net = models.segmentation.fcn_resnet50(pretrained=False,
    #                                       num_classes=2,
    #                                       pretrained_backbone=False)

    # Have ResNet model take in grayscale rather than RGB
    # net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Convert batch norm layers to group norm
    # net = nets.convertBNtoGN(net)

    # This construct allows us to test different loss functions by passing them in as part of the config parameter
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda(), ignore_index=255)
    if config["loss"] == "IOU":
        criterion = losses.IoULoss()
    elif config["loss"] == "Dice":
        criterion = losses.DiceBCELoss()
    elif config["loss"] == "Focal:":
        criterion = losses.FocalLoss()
    elif config["loss"] == "BCE":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1]).float().cuda(), ignore_index=255)


    net = nets.UNET(2, 2)
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(net.parameters(), weight_decay=config["wd"], lr=LR, betas=(config["b1"], config["b2"]))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2,
                                                                     eta_min=0,
                                                                     last_epoch=-1)
    # Check if graphics card is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Put net on whatever device is available
    net = net.to(device)

    # This block allows ray tune to save checkpoints
    # if checkpoint_dir:
    #      model_state, optimizer_state = torch.load(
    #          os.path.join(checkpoint_dir, "checkpoint"))
    #      net.load_state_dict(model_state)
    #      optimizer.load_state_dict(optimizer_state)

    #Create tensorboard log
    log_name = "./runs/" + MODEL_NAME
    writer = SummaryWriter(log_dir=log_name, comment=config)
    best_iou = 0.0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        #Train loop
        train_dict = train_loop(train_loader, net, criterion, optimizer)
        #Log train loop outputs
        writer.add_scalar("Loss/Train", train_dict["Train_Loss"], epoch + 1)
        writer.add_scalar("IOU/Train", train_dict["Train_IOU"], epoch + 1)
        writer.add_scalar("ACC/Train", train_dict["Train_Acc"], epoch + 1)
        #Valid loop
        valid_dict = valid_loop(valid_loader, net, criterion)
        #Log valid loop outputs
        writer.add_scalar("Loss/Valid", valid_dict["Valid_Loss"], epoch + 1)
        writer.add_scalar("IOU/Valid", valid_dict["Valid_IOU"], epoch + 1)
        writer.add_scalar("ACC/Valid", valid_dict["Valid_Acc"], epoch + 1)
        #Ray tune checkpointing remnant. Uncomment if working with ray tune
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)
        # tune.report(iou=valid_dict["Valid_IOU"], acc=valid_dict["Valid_Acc"], loss=valid_dict["Valid_Loss"])
        if epoch % 10 == 0:
            torch.save(net.state_dict(), MODEL_NAME + str(epoch) + ".pth")
            print(f"Saving Checkpoint. {valid_dict}")
        scheduler.step()
    print('Finished Training')
    writer.flush()
    writer.close()
    #torch.save(net.state_dict(), MODEL_NAME)
    print(f"Final Model Saved")


# Function for hyperparameter tuning using the Ray library. Takes a config dictionary and runs different variations
# of the neural network training function train_sen, terminating unsuccessful configurations and saving successful ones.
def ray_tune(num_samples=18, max_num_epochs=12, gpus_per_trial=1):
    data_dir = Path("W:/cloud2street/flood_nn/")
    checkpoint_dir = Path("W:/cloud2street/flood_nn/checkpoints/")
    config = {
        "loss": tune.choice(["IOU", "Focal", "Dice"]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "bs": tune.choice([8, 16, 32])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)
    result = tune.run(
        tune.with_parameters(train_sen, data_dir=data_dir),
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
        name="unet_test",
        local_dir="W:/cloud2street/flood_nn/ray_results"
    )


if __name__ == '__main__':
    #These are hyperparameter training using Ray tune remnants left in in case I need to use them. Ray Tune on windows
    #is kind of unstable, so I implemented a simpler version just using a config dictionary for longer epoch testing.
    # ray_tune()
    # train_sen(data_dir=Path("W:/cloud2street/flood_nn/"))
    # runs_list = [
    #     {"epochs": 100, "lr": .00988, "loss": "Focal", "dataset": "S2Weak", "model_name": "S2weak_focal_unet_00988"},
    #     {"epochs": 100, "lr": .0005, "loss": "BCE", "dataset": "S2Weak", "model_name": "S2weak_BCE_unet_0005"},
    #     {"epochs": 100, "lr": .00063, "loss": "Dice", "dataset": "S2Weak", "model_name": "S2weak_Dice_unet_00063"},
    #     {"epochs": 100, "lr": .00407, "loss": "IOU", "dataset": "S1Weak", "model_name": "S1weak_IOU_unet_00407"},
    # ]
    # runs_list = [
    #     {"epochs": 50, "lr": .0005, "loss": "BCE", "dataset": "S1Weak", "model_name": "S1weak_BCE_unet_0005", "bs" : 16},
    #     {"epochs": 50, "lr": .00063, "loss": "Dice", "dataset": "S1Weak", "model_name": "S1weak_Dice_unet_00063", "bs" : 16},
    # ]
    # runs_list = [
    #     {"epochs": 50, "lr": .0005, "loss": "BCE", "dataset": "S1Weak", "model_name": "S1weak_BCE_unet_0005_bs32", "bs" : 32},
    #     {"epochs": 50, "lr": .0005, "loss": "BCE", "dataset": "S2Weak", "model_name": "S2weak_BCE_unet_0005_bs32",
    #      "bs": 32}
    # ]
    runs_list = [
        {"epochs": 400, "lr": .0005, "loss": "BCE", "dataset": "S1Hand", "model_name": "S1hand_BCE_unet_0005_bs16_betas2to66_wd03",
         "bs": 16, "b1": .2, "b2" : 0.66, "wd": .003},
    ]
    for config in runs_list:
        train_sen(data_dir=Path("W:/cloud2street/flood_nn/"), config=config)

import torch
from pathlib import Path
import numpy as np
import csv
import flooddata as fl
import torch.utils.data
import matplotlib.pyplot as plt
import nets
import tests as t

# Loads list of file paths for use by the dataset. Returns a list formatted [training paths, valid paths, test paths]
import tests


def get_data_splits(data_type):
    train_path = Path("./splits/" + data_type + "/train_split.csv")
    valid_path = Path("./splits/" + data_type + "/valid_split.csv")
    test_path = Path("./splits/" + data_type + "/test_split.csv")
    path_list = [train_path, valid_path, test_path]
    out_list = []
    for path in path_list:
        file_list = []
        with open(path, newline='') as csvfile:
            fileread = csv.reader(csvfile, delimiter=',')
            for row in fileread:
                to_append = Path(row[0]), Path(row[1])
                file_list.append(to_append)
        out_list.append(file_list)
    return out_list


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    valid_iou, valid_acc, fp, fn, tn, tp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    count = 0
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for images, labels, item in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            # Uncomment to use with resnet based nets
            # pred = model(images)['out'].cuda()
            pred = model(images)
            count += 1
            valid_iou += t.computeIOU(pred, labels)
            valid_acc += t.computeAccuracy(pred, labels)
            fp += t.falsePositives(pred, labels)
            fn += t.falseNegatives(pred, labels)
            tn = t.trueNegatives(pred, labels)
            tp = t.truePositives(pred, labels)
    return {"Test_IOU": valid_iou / count, "Test_Acc": valid_acc / count, "Test_Comis": fp/(fp+tp), "Test_Omis": fn/(fn+tn)}

def load_net(file_path):
    # Load our pretrained net. Depending on type of net we have to do this in different ways, hence the commented portions
    # net = models.segmentation.fcn_resnet50(pretrained=False,
    #                         num_classes=2,
    #                          pretrained_backbone=False)

    # net.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # net = nets.convertBNtoGN(net)

    net = nets.UNET(2, 2)

    # Load saved state dictionary
    net.load_state_dict(torch.load(Path(file_path)))
    # Set net to evaluation mode and transfer it to the graphics card
    net.eval()
    net.to("cuda")
    return net


def create_images(type, set, file_number):
    net = load_net("S1weak_BCE_unet_0005_bs16_betas2to66_wd0380.pth")
    # Load a set of test data into a PyTorch dataset
    if type == "Bolivia":
        test = get_data_list(Path("W:/cloud2street/flood_nn/files\data/flood_events/HandLabeled/Bolivia_Hand_Image"),
                            Path("W:/cloud2street/flood_nn/files\data/flood_events/HandLabeled/Bolivia_Hand_Label"))
    else:
        test = get_data_splits(type)[set]
    test_data = fl.FloodValidData(test)
    # Pick out a sample from the dataset
    base, label, item = test_data[file_number]
    # Our nets expect a item of 3 dimensions since we worked in batches, this adds a dummy dimension
    base = base.unsqueeze(dim=0)
    label = label.unsqueeze(dim=0)
    # Send it to the graphics card
    base = base.to("cuda")
    label = label.to("cuda")
    label = label.long()
    # Get a predicted set of water labels from our model
    with torch.no_grad():
        predict = net(base)
        predict = torch.argmax(predict, dim=1)
    # Remove the dummy dimensions
    predict = predict.squeeze(dim=0)
    base = base.squeeze(dim=0)
    label = label.squeeze(dim=0)
    # Convert everything to plottable images
    p = np.uint8(predict.cpu()) * 255
    c1 = np.uint8(base[0, :, :].cpu()) * 255
    c2 = np.uint8(base[1, :, :].cpu()) * 255
    lab = np.uint8(label.cpu()) * 255

    # Make a nice plot showing inputs, predicted output, and provided output
    plt.rcParams["figure.figsize"] = (12, 12)
    fig, ax = plt.subplots(2,2)
    ax[0, 0].imshow(c1, cmap="gist_earth")
    ax[1, 0].imshow(c2, cmap="gist_earth")
    ax[0, 1].imshow(p, cmap="Blues")
    ax[1, 1].imshow(lab, cmap="Blues")

    ax[0, 0].set_title("Original Image Channel 1")
    ax[1, 0].set_title("Original Image Channel 2")
    ax[0, 1].set_title("Predicted Label (ML)")
    #Change this label depending on what type of dataset is being used
    ax[1, 1].set_title("Hand Label")
    fig.suptitle(item[0] + "\nIoU: " + str(tests.computeIOU(predict, label, argmax=False)))
    plt.show()


# Tests an entire dataset for mean IOU and accuracy
def test_whole_dataset(type):
    if type == "Bolivia":
        set = get_data_list(Path("W:/cloud2street/flood_nn/files\data/flood_events/HandLabeled/Bolivia_Hand_Image"), Path("W:/cloud2street/flood_nn/files\data/flood_events/HandLabeled/Bolivia_Hand_Label"))
    else:
        set = get_data_splits(type)
        set = set[0] + set[1] + set[2]
    net = load_net("W:/cloud2street/flood_nn/S1weak_BCE_unet_0005_bs16_betas2to66_wd0380.pth")
    test_data = fl.FloodValidData(set)
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=4,
                                               pin_memory=True,
                                               num_workers=4)
    test_results = test_loop(test_loader, net)
    return test_results

# Return a paired list of all files in imagery and label folders. Just using this here to access the Bolivia data.
def get_data_list(imagery_folder, label_folder):
    file_list = []
    # Check the paths then get images and labels from folders
    if label_folder.exists() & imagery_folder.exists():
        # Convert generators to lists, ensure they are sorted the same way since generator output is arbitrary.
        images = sorted(list(imagery_folder.iterdir()))
        labels = sorted(list(label_folder.iterdir()))

        # Create a list of image,label tuples
        file_list = [(images[x], labels[x]) for x in range(0, len(images))]
    else:
        message = "Files not found. Check your folder structure"
        return print(message)
    return file_list


# On Windows many PyTorch functions have to be hidden within this or else there will be problems with multithreading. With
# more time I could make this a nice command line utility
if __name__ == "__main__":
    create_images("Bolivia", 2, 1)
    #print(test_whole_dataset("Bolivia"))

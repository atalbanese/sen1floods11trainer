import torch

# ASSESSMENT FUNCTIONS. Adapted from the Sen1Floods11 researchers.
def computeIOU(output, target, argmax=True):
    if argmax:
        output = torch.argmax(output, dim=1)
    output = output.flatten()
    target = target.flatten()

    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    intersection = torch.sum(output * target)
    union = torch.sum(target) + torch.sum(output) - intersection
    iou = (intersection + .0000001) / (union + .0000001)
    iou = iou.to("cpu")
    if iou != iou:
        print("failed, replacing with 0")
        iou = torch.tensor(0).float()

    return iou


def computeAccuracy(output, target, argmax=True):
    if argmax:
        output = torch.argmax(output, dim=1)
    output = output.flatten()
    target = target.flatten()

    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output.eq(target))
    correct = correct.to("cpu")
    return correct.float() / len(target)


def truePositives(output, target, argmax=True):
    if argmax:
        output = torch.argmax(output, dim=1)
    output = output.flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output * target)

    return correct.to("cpu")


def trueNegatives(output, target, argmax=True):
    if argmax:
        output = torch.argmax(output, dim=1)
    output = output.flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 0)
    target = (target == 0)
    correct = torch.sum(output * target)

    return correct.to("cpu")


def falsePositives(output, target, argmax=True):
    if argmax:
        output = torch.argmax(output, dim=1)
    output = output.flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 1)
    target = (target == 0)
    correct = torch.sum(output * target)

    return correct.to("cpu")


def falseNegatives(output, target, argmax=True):
    if argmax:
        output = torch.argmax(output, dim=1)
    output = output.flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 0)
    target = (target == 1)
    correct = torch.sum(output * target)

    return correct.to("cpu")
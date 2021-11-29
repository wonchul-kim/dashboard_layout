import torchvision.models as models 

def torchvision_models(model_name, pretrained, loss, num_classes):
    if loss == 'aux_loss':
        model = models.segmentation.__dict__[model_name](pretrained=pretrained,
                                                                progress=True,
                                                                aux_loss=True)
    else:
        model = models.segmentation.__dict__[model_name](pretrained=pretrained,
                                                                progress=True,
                                                                aux_loss=False)
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    return model
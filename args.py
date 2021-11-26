class args():
    # training args
    epochs = 2  # "number of training epochs, default is 2"
    batch_size = 4  # "batch size for training, default is 4"
    # data_set
    # URL:
    data_set = "E:/COCO_training81783_gray_256-256"
    # data_set = "E:/Dataset/train2014COCO/COCO3000"
    HEIGHT = 256
    WIDTH = 256
    downsample = ['stride', "avgpool", "maxpool"]

    ch = [8, 16, 32, 64, 128, 256, 512, 1024]  # channels
    save_model_dir_encoder = "models/fusion_FusionModule"
    save_loss_dir = "D:/wjy/multi scale dense/models/loss_FusionModule"

    cuda = 1
    ssim_weight = [1, 10, 100, 1000, 10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']
    grad_weight = [1, 10, 100, 1000, 10000]

    lr = 1e-3  # "learning rate, default is 0.001"
    lr_light = 1e-4  # "learning rate, default is 0.001"
    log_interval = 10  # "number of images after which the training loss is logged, default is 500"
    resume = None

    # for test, model_default is the model used in paper
    model_default = 'UNFusion.model'
    model_deepsuper = 'UNFusion.model'
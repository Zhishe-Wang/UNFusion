import os
import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import FusionModule
from args import args
import pytorch_msssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    original_imgs_path = utils.list_images(args.data_set)
    train_num = 81783
    original_imgs_path = original_imgs_path[:train_num]
    random.shuffle(original_imgs_path)
    for i in range(2, 3):
        train(i, original_imgs_path)


def train(i, original_imgs_path):
    batch_size = args.batch_size
    Is_testing = 0

    UNF_model = FusionModule(Is_testing)

    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        UNF_model.load_state_dict(torch.load(args.resume))
    print(UNF_model)
    optimizer = Adam(UNF_model.parameters(), args.lr)
    L1_loss = torch.nn.L1Loss(reduction="mean")
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        UNF_model.cuda()

    tbar = trange(args.epochs)
    print('Start training.....')

    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_pixel_loss = 0.
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        UNF_model.train()
        count = 0
        for batch in range(batches):
            image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)
            count += 1
            optimizer.zero_grad()
            img = Variable(img, requires_grad=False)
            if args.cuda:
                img = img.cuda()
            # get fusion image
            # encoder
            en = UNF_model.encoder(img)
            # decoder
            output = UNF_model.decoder(en, Is_testing)
            # resolution loss: between fusion image and visible image
            x = Variable(img.data.clone(), requires_grad=False)

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            pixel_loss_temp = L1_loss(output, x)
            ssim_loss_temp = ssim_loss(output, x, normalize=True)
            ssim_loss_value += (1 - ssim_loss_temp)
            pixel_loss_value += pixel_loss_temp
            ssim_loss_value /= len(output)
            pixel_loss_value /= len(output)

            # total loss
            a = 0.87
            total_loss = (1-a) * pixel_loss_value + a * ssim_loss_value
            total_loss.backward()
            optimizer.step()

            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\t Epoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), e + 1, count, batches,
                                     all_pixel_loss / args.log_interval,
                                     (all_ssim_loss) / args.log_interval,
                                     (a * all_ssim_loss + (1-a) * all_pixel_loss) / args.log_interval
                )
                tbar.set_description(mesg)
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_all.append((a * all_ssim_loss + (1-a) * all_pixel_loss) / args.log_interval)
                count_loss = count_loss + 1
                all_ssim_loss = 0.
                all_pixel_loss = 0.

            if (batch + 1) % (200 * args.log_interval) == 0:
                # save model
                UNF_model.eval()
                UNF_model.cpu()
                save_model_path = "Epoch_" + str(e) + "_iters_" + str(count) + "_" + args.ssim_path[
                                          i] + ".model"
                torch.save(UNF_model.state_dict(), save_model_path)
                # save loss data
                # pixel loss
                loss_data_pixel = Loss_pixel
                loss_filename_path = "loss_pixel_epoch_" + ".mat"
                scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
                # SSIM loss
                loss_data_ssim = Loss_ssim
                loss_filename_path = "loss_ssim_epoch_" + ".mat"
                scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
                # all loss
                loss_data = Loss_all
                loss_filename_path ="loss_all_epoch.mat"
                scio.savemat(loss_filename_path, {'loss_all': loss_data})

                UNF_model.train()
                UNF_model.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

    # pixel loss
    loss_data_pixel = Loss_pixel
    loss_filename_path = '001.mat'
    scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
    loss_data_ssim = Loss_ssim
    loss_filename_path = "002.mat"
    scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
    # SSIM loss
    loss_data = Loss_all
    loss_filename_path = "003.mat"
    scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
    # save model
    UNF_model.eval()
    UNF_model.cpu()
    save_model_filename = "001.model"
    torch.save(UNF_model.state_dict(), save_model_filename)

    print("\nDone, trained model saved at", save_model_filename)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

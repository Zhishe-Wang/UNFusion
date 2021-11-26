import os
import torch
from torch.autograd import Variable
from net import FusionModule
import utils
from args import args
import numpy as np
import time


def load_model(path):
    Is_testing = 1
    UNF_model = FusionModule(Is_testing)
    UNF_model.load_state_dict(torch.load(path))
    print(UNF_model)

    para = sum([np.prod(list(p.size())) for p in UNF_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(UNF_model._get_name(), para * type_size / 1000 / 1000))

    UNF_model.eval()
    UNF_model.cuda()

    return UNF_model


def run_demo(UNF_model, infrared_path, visible_path, output_path_root, index, f_type):
    Is_testing = 1
    img_ir, h, w, c = utils.get_test_image(infrared_path)
    img_vi, h, w, c = utils.get_test_image(visible_path)

    # dim = img_ir.shape
    if args.cuda:
        img_ir = img_ir.cuda()
        img_vi = img_vi.cuda()
    img_ir = Variable(img_ir, requires_grad=False)
    img_vi = Variable(img_vi, requires_grad=False)
    # encoder
    en_ir = UNF_model.encoder(img_ir)
    en_vi = UNF_model.encoder(img_vi)
    # fusion
    f = UNF_model.fusion(en_ir, en_vi, f_type)
    # decoder
    img_fusion_list = UNF_model.decoder(f, Is_testing)

    for img_fusion in img_fusion_list:
        str_index = str(index + 1000)
        file_name = "UNFusionnet_"+f_type +"_" + str_index + '.png'
        output_path = output_path_root + file_name
        # save images
        utils.save_image_test(img_fusion, output_path)
        print(output_path)


def main():
   fusion_type = ['l1_mean', 'l2_mean', 'linf']

   with torch.no_grad():
       model_path = args.model_default
       model = load_model(model_path)
       for j in range(len(fusion_type)):
           start = time.time()
           output_path = fusion_type[j]

           if os.path.exists(output_path) is False:
               os.mkdir(output_path)
           output_path = output_path + '/'

           f_type = fusion_type[j]
           print('Processing......  ' + f_type)

           ir = utils.list_images("G:/UNFusionnet\IVs_images/thermal")
           vis = utils.list_images("G:/UNFusionnet\IVs_images/visual")

           for i in range(25):
               index = i + 1
               infrared_path = ir[i]
               visible_path = vis[i]

               run_demo(model, infrared_path, visible_path, output_path, index, f_type)
           end = time.time()
           print("Testing success,Testing time is [%f]" % (end - start))

   print('Done......')


if __name__ == '__main__':
   main()

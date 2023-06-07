import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from pathlib import Path




if __name__=="__main__":
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    results_dir =  Path(opt.dataroot).parents[1]
    #image_dir = os.path.join(opt.results_dir, opt.name)
    
    image_dir = os.path.join(results_dir)

    # test
    if not opt.engine and not opt.onnx:
        print("[Test.py] Marker 1")
        model = create_model(opt)
        print("[Test.py] Marker 2")
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx
        
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                            opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1 
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:        
            generated = model.inference(data['label'], data['inst'], data['image'])
            
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                            ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        print(f'[INFO] Processing image: {i+1}', end='\r')
        visualizer.save_images_xyz(image_dir, visuals, img_path)
    print("[INFO] Done processing images via test.py")
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from collections import defaultdict
import torch
import numpy as np
import os, sys
sys.path.append(os.path.join(os.getcwd(), "src/mirracle_multimodal/mirracle_multimodal"))
import models
import argparse

parser = argparse.ArgumentParser(description='Fusion inference args.')
parser.add_argument('--pth', default=None,
                    help='folder with the model to infer with')
parser.add_argument('--data', default=None,
                    help='path to the folder with training data (for offline testing)')
args = parser.parse_args()


def load_images(path, imsize=64):
        import os, glob, numpy as np, imageio
        images = sorted(glob.glob(os.path.join(path, "*.png")))
        dataset = np.zeros((len(images), imsize, imsize, 3), dtype=np.float)
        for i, image_path in enumerate(images):
            image = imageio.imread(image_path)
            # image = reshape_image(image, self.imsize)
            # image = cv2.resize(image, (imsize, imsize))
            dataset[i, :] = image / 255
        print("Dataset of shape {} loaded".format(dataset.shape))
        return dataset

def load_model(path):
    args = torch.load(os.path.join(path, 'args.rar'))
    device = torch.device("cuda" if args.cuda else "cpu")
    model = str(args.modalities_num) if args.modalities_num == 1 else "_".join((str(args.modalities_num), args.mixing))
    modelC = getattr(models, 'VAE_{}'.format(model))
    model = modelC(vars(args)).to(device)
    print('Loading model {} from {}'.format(model.modelName, path))
    model.load_state_dict(torch.load(os.path.join(path, 'model.rar')))
    model._pz_params = model._pz_params
    model.eval()
    return model

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'fuse_input', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.dataset = None
        if args.data:
            self.dataset = self.load_dataset()

    def load_dataset(self):
        d = load_images(args.data)
        d = d.reshape((d.shape[0], 12288))
        print("Dataset loaded")
        return d

    def timer_callback(self):
        if self.dataset:
            d1 = self.dataset[self.i].tolist()
        else:
            d1 = [v for v in np.random.rand(12288)]
        msg = Float32MultiArray()
        msg.data = d1
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing data; iteration %s' % self.i)
        self.i += 1

class FusePublisher(Node):
    def __init__(self):
        super().__init__('inference_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'fuse_infer', 1)

    def publish_data(self, data):
        msg = Float32MultiArray()
        msg.data = data
        self.publisher_.publish(msg)
        self.get_logger().info('data fused and published')


class InputFuseSubscriber(Node):
    def __init__(self, model):
        super().__init__('minimal_subscriber')
        self.publisher = FusePublisher()
        self.model = model
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'fuse_input',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Initiated, waiting for input data")

    def listener_callback(self, msg):
        d = np.asarray(msg.data)
        d1 = d[:12288].reshape(3, 64, 64)
        d1 = torch.tensor(d1).unsqueeze(0)
        d2 = d[12288:].reshape(12288)
        d2 = torch.tensor(d2).unsqueeze(0)
        qz_x, px_z, zs = self.model([d1,d2], 1)
        data = np.asarray(px_z[0].loc.squeeze().detach().cpu()).flatten().tolist()
        self.publisher.publish_data(data)


def main_fusion():
    if not args.pth:
        print("Missing path to the pretrained inference model folder. Run again with --pth argument")
        raise
    model = load_model(args.pth)
    rclpy.init()
    inferator = InputFuseSubscriber(model)
    rclpy.spin(inferator)
    inferator.destroy_node()
    rclpy.shutdown()

def main_dummydata(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main_dummydata()

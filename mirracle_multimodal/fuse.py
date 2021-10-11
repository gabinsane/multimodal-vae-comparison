import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
import numpy as np

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'fuse_input', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        d1 = [v for v in np.random.rand(12288*2)]
        d2 = np.random.rand(1,12288)
        #data = np.concatenate((d1,d2))
        msg = Float32MultiArray()
        msg.data = d1
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing image and text; iteration %s' % self.i)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    #minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_publisher)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    #minimal_subscriber.destroy_node()
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
from torch.utils.tensorboard import SummaryWriter

class log_writer():
    def __init__(self, log_path, describtion):
        self.writer = SummaryWriter(log_path, 'Training process of cars detection via ResNet18!')

    def add_graph(self, image):
        self.writer.add_graph()

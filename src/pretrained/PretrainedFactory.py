from src.pretrained.ResNet50 import ResNet


class PretrainedFactory:
    def __init__(self, conf):
        self.conf = conf

    def resnet50(self):
        return ResNet(self.conf)

    def get_pretrained_model(self, argument):
        switcher = {
            "ResNet50": self.resnet50()
        }
        pretrained_model = switcher.get(argument, lambda: "Invalid option")
        return pretrained_model

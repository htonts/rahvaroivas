import os

import tensorflow as tf
from imageai.Classification.Custom import ClassificationModelTrainer
from datetime import datetime
from datetime import timedelta

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)


def train(data_directory, models_subdirectory, num_objects):
    start = datetime.now().strftime("%H:%M:%S")
    model_trainer = ClassificationModelTrainer()

    if "InceptionV3" in models_subdirectory:
        model_trainer.setModelTypeAsInceptionV3()
    elif "MobileNetV2" in models_subdirectory:
        model_trainer.setModelTypeAsMobileNetV2()
    elif "ResNet50" in models_subdirectory:
        model_trainer.setModelTypeAsResNet50()
    elif "DenseNet121" in models_subdirectory:
        model_trainer.setModelTypeAsDenseNet121()

    model_trainer.setDataDirectory(data_directory=data_directory,
                                   models_subdirectory=models_subdirectory)
    model_trainer.trainModel(num_objects=num_objects,
                             num_experiments=100,
                             enhance_data=True,
                             batch_size=32)
    end = datetime.now().strftime("%H:%M:%S")
    start_to_end = datetime.strptime(end, "%H:%M:%S") - datetime.strptime(start, "%H:%M:%S")
    if start_to_end.days < 0:
        start_to_end = timedelta(days=0,
                                 seconds=start_to_end.seconds,
                                 microseconds=start_to_end.microseconds)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_file = open("logs/trainingLogs.txt", "a")
    log_file.write(
        "\nTime \"" + data_directory + "\" " + models_subdirectory + " model training was started: " + start +
        "\nTime \"" + data_directory + "\" " + models_subdirectory + " model training was finished: " + end +
        "\nTime it took to complete model training: " + str(start_to_end))
    log_file.close()


train("regioonidSeelikud", "InceptionV3", 4)
train("regioonidSeelikud", "MobileNetV2", 4)
train("regioonidSeelikud", "ResNet50", 4)
train("regioonidSeelikudVööd", "InceptionV3", 4)
train("regioonidSeelikudVööd", "MobileNetV2", 4)
train("regioonidSeelikudVööd", "ResNet50", 4)
train("maakonnadSeelikud", "InceptionV3", 14)
train("maakonnadSeelikud", "MobileNetV2", 14)
train("maakonnadSeelikud", "ResNet50", 14)
train("maakonnadSeelikudVööd", "InceptionV3", 14)
train("maakonnadSeelikudVööd", "MobileNetV2", 14)
train("maakonnadSeelikudVööd", "ResNet50", 14)
train("valladSeelikud", "InceptionV3", 47)
train("valladSeelikud", "MobileNetV2", 47)
train("valladSeelikud", "ResNet50", 47)
train("valladSeelikudVööd", "InceptionV3", 48)
train("valladSeelikudVööd", "MobileNetV2", 48)
train("valladSeelikudVööd", "ResNet50", 48)

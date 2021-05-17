from imageai.Prediction.Custom import CustomImageClassification
import os
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config)

prediction = CustomImageClassification()
model = "ResNet50-RSV_A-0.967625.h5"
folder_path = "testpildid"
execution_path = os.getcwd() + "\\regioonidSeelikudVööd\\"

if "InceptionV3" in model:
    prediction.setModelTypeAsInceptionV3()
elif "MobileNetV2" in model:
    prediction.setModelTypeAsMobileNetV2()
elif "ResNet50" in model:
    prediction.setModelTypeAsResNet50()

prediction.setModelPath(os.path.join(os.getcwd(), "mudelid\\regioonid\\" + model))
prediction.setJsonPath(os.path.join(execution_path, "json\\model_class.json"))
prediction.loadModel(num_objects=4)

log_file = open("logs\\results.txt", "a")
log_file.write("Testing model " + model + " on path " + execution_path + "...\n")
for paths, _, folders in os.walk(folder_path):
    for file in folders:
        file = os.path.abspath(os.path.join(paths, file))
        predictions, probabilities = prediction.classifyImage(file, result_count=4)
        result = ""
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            result = result + eachPrediction + " : " + str(eachProbability) + "%\n"
        log_file.write(file + "\n")
        log_file.write(result)
log_file.write("Testing complete!\n")
log_file.close()


args = dict()

# data
args["CODE_DIR"] = "./" 
args["SAVED_WEIGHTS_DIR"] = "./weights"
args["DATASET_DIR"] = "./"
args["DATA_DIR"] = '/content/drive/MyDrive/Datasets/Thesis/AV-RelScore/datasets'
args["PRETRAINED_MODEL"] =  None

# train
args["USE_MAN_SEED"] = False # Whether use manual seed or not
args["SEED"] = 72323
args["IMAGE_SIZE"] = (256, 256)
args["CRITIC_UPDATE"] = 5 # Number of iterations of critical model in each batch iteration
args["BATCH_SIZE"] = 4 
args["EPOCHS"] = 50
args["SAVE_FREQ"] = 5 #saving the model weights and loss/metric plots after every these many steps

#optimizer and scheduler
args["INIT_LR"] = 1e-4  #initial learning rate for scheduler
args["FINAL_LR"] = 1e-11 #final learning rate for scheduler
args["WEIGHT_DECAY"] = 0.5   #learning rate decrease factor for scheduler
args["MOMENTUM1"] = 0.9 #optimizer momentum 1 value
args["MOMENTUM2"] = 0.999   #optimizer momentum 2 value
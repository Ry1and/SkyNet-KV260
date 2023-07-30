from utils import *
from models import *

modeltype = 'SkyNet()'
weightfile = 'dac.weights'

model = eval(modeltype)
load_net(weightfile, model)


device = 'cpu'
#device = 'cuda'

model = model.to(device)
model.eval()

print("Ready to run.")

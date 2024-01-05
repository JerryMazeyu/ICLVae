from runner.runners import *
from src.model import llama2
from src.data import twitter_data, sst2_data, cola_data
from configs import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# runner = SentimentClassificationTrainer(glm, twitter_data, baseconf)
# runner.run()

# runner = FeatureMapVisiualizeRunner(glm4v, twitter_data, visiualconf)
# runner.run()

# runner = ValidateAttitudeRunner(glm4v, twitter_data, visiualconf)
# runner.run()

# runner = ValidateAttitudeRunner(glm4v, sst2_data, visiualconf)
# runner.run()

runner = GrammaticalAcceptableTrainer(llama2, cola_data, baseconf)
runner.run()
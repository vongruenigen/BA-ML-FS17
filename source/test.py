from config import Config
from model import Model
from runner import Runner

cfg = Config()
runner = Runner(cfg)
runner.train()

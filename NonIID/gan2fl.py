import numpy as np
import pandas as pd



class client():
    def __init__(self,cid,net,trainset,testset):
        self.cid=cid 
        self.model=net
        self.trainset=trainset
        self.testset=testset


parser = argparse.ArgumentParser(description="Flower")
import random
import argparse
import torch.optim as optim

parser = argparse.ArgumentParser(description='Model parameters')
parser.add_argument('discount', default=0.8, type=float, help='Discount value (gamma) for Q-function')
parser.add_argument('epsilon', default=0.7, type=float, help='Probability of taking random action')
parser.add_argument('batch', default=50, type=int, help='Minibatch size')
parser.add_argument('observe', default=500, type=int, help='Observe result after N timesteps')
parser.add_argument('learning_rate', default=0.01, type=float, help='Learning rate of the DQN')
args = parser.parse_args()

def main(){
    opt = {}
    opt['discount'] = args.discount
    opt['epsilon'] = args.epsilon
    opt['batch_size'] = args.batch
    opt['observe_freq'] = args.observe
    opt['learning_rate'] = args.learning_rate

    model = ParkingAgent()
}

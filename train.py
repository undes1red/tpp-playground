from src import TrainingHost
import os

# Hope we can get rid of absolute path in training scripts.
root_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # Train 
    agent = TrainingHost(root_path = root_path, procedure_name = 'TPP')
    agent.start()
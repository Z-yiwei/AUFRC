"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from skipganomaly import Skipganomaly
##

##
def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)
    model = Skipganomaly(opt, data)
    model.train()

if __name__ == '__main__':
    main()

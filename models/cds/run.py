import sys
import os
import torch
import logging
import platalea.basic as M
import platalea.dataset as D

seed=int(sys.argv[1])
torch.manual_seed(seed)

logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
root = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + "/data/NewmanRatner/"
data = dict(train=D.NewmanRatner_loader(split='train', register="CDS", root=root, batch_size=16, shuffle=True),
            val=D.NewmanRatner_loader(split='val', register="CDS", root=root, batch_size=16, shuffle=False))

config = dict(SpeechEncoder=dict(conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                                 rnn= dict(input_size=64, hidden_size=1024, num_layers=4, bidirectional=True, dropout=0),
                                 att= dict(in_size=2048, hidden_size=128)),
              ImageEncoder=dict(linear=dict(in_size=768, out_size=2*1024), norm=True),
              margin_size=0.2)

logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=50, seed=seed)

logging.info('Training')
folder = os.path.dirname(os.path.abspath(__file__))
M.experiment(net, data, run_config, folder=folder)

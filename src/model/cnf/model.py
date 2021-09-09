import torch
from ..utils import BasicModule

from .models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, JumpGMMSpatiotemporalModel
from .models.spatial import GaussianMixtureSpatialModel, IndependentCNF, JumpCNF, SelfAttentiveCNF
from .models.temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess, NeuralPointProcess
from .models.temporal.neural import ACTFNS as TPP_ACTFNS


class CNFWrapper(BasicModule):
    def __init__(self, device, model_type, tpp_type, **kwargs):
        '''
        CNF model wrapper
        '''
        super(CNFWrapper, self).__init__()

        self.device = device
        # I'm wondering how to get these numbers.
        self.t_0 = torch.tensor([0.0], device = self.device)
        self.t_1 = torch.tensor([250.0], device = self.device)
        
        if kwargs.get('tpp_model') is None:
            if kwargs['tpp_actfn'] not in TPP_ACTFNS.keys():
                raise Exception(f"Invalid tpp model activation function. Available activations are {TPP_ACTFNS.keys()}")
        else:
            if kwargs['tpp_model']['tpp_actfn'] not in TPP_ACTFNS.keys():
                raise Exception(f'Invalid tpp model activation function. Available activations are {TPP_ACTFNS.keys()}')

        if model_type == 'jumpcnf' and tpp_type == 'neural':
            self.model = JumpCNFSpatiotemporalModel(**kwargs).to(self.device)
            # dim=x_dim,
            # hidden_dims=list(map(int, args.hdims.split("-"))),
            # tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
            # actfn=args.actfn,
            # tpp_cond=args.tpp_cond,
            # tpp_style=args.tpp_style,
            # tpp_actfn=args.tpp_actfn,
            # share_hidden=args.share_hidden,
            # solve_reverse=args.solve_reverse,
            # tol=args.tol,
            # otreg_strength=args.otreg_strength,
            # tpp_otreg_strength=args.tpp_otreg_strength,
            # layer_type=args.layer_type,
        elif model_type == 'attncnf' and tpp_type == 'neural':
            self.model = SelfAttentiveCNFSpatiotemporalModel(**kwargs).to(self.device)
            # dim=x_dim
            # hidden_dims=list(map(int, args.hdims.split("-"))),
            # tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
            # actfn=args.actfn,
            # tpp_cond=args.tpp_cond,
            # tpp_style=args.tpp_style,
            # tpp_actfn=args.tpp_actfn,
            # share_hidden=args.share_hidden,
            # solve_reverse=args.solve_reverse,
            # l2_attn=args.l2_attn,
            # tol=args.tol,
            # otreg_strength=args.otreg_strength,
            # tpp_otreg_strength=args.tpp_otreg_strength,
            # layer_type=args.layer_type,
            # lowvar_trace=not args.naive_hutch,
        elif model_type == 'cond_gmm' and tpp_type == 'neural':
            self.model = JumpGMMSpatiotemporalModel(**kwargs).to(self.device)
            # dim=x_dim,
            # hidden_dims=list(map(int, args.hdims.split("-"))),
            # tpp_hidden_dims=list(map(int, args.tpp_hdims.split("-"))),
            # actfn=args.actfn,
            # tpp_cond=args.tpp_cond,
            # tpp_style=args.tpp_style,
            # tpp_actfn=args.tpp_actfn,
            # share_hidden=args.share_hidden,
            # tol=args.tol,
            # tpp_otreg_strength=args.tpp_otreg_strength,
        else:        # Mix and match between spatial and temporal models.
            if tpp_type == "poisson":
                tpp_model = HomogeneousPoissonPointProcess()
            elif tpp_type == "hawkes":
                tpp_model = HawkesPointProcess()
            elif tpp_type == "correcting":
                tpp_model = SelfCorrectingPointProcess()
            elif tpp_type == "neural":
                tpp_model = NeuralPointProcess(**kwargs['tpp_model'])
                # cond_dim=x_dim, hidden_dims=list(map(int, args.tpp_hdims.split("-"))), cond=args.tpp_cond, style=args.tpp_style, actfn=args.tpp_actfn,
                # otreg_strength=args.tpp_otreg_strength, tol=args.tol)
            else:
                raise ValueError(f"Invalid tpp model {tpp_type}")
    
            if model_type == "gmm":
                self.model = CombinedSpatiotemporalModel(GaussianMixtureSpatialModel(), tpp_model).to(self.device)
            elif model_type == "cnf":
                self.model = CombinedSpatiotemporalModel(IndependentCNF(**kwargs['model']), tpp_model).to(self.device)
                    # dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                    # layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength,
                    # squash_time=True
            elif model_type == "tvcnf":
                self.model = CombinedSpatiotemporalModel(IndependentCNF(**kwargs['model']), tpp_model).to(self.device)
                    # dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                    # layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
                    # tpp_model
            elif model_type == "jumpcnf":
                self.model = CombinedSpatiotemporalModel(JumpCNF(**kwargs['model']), tpp_model).to(self.device)
                    # dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                    # layer_type=args.layer_type, actfn=args.actfn, tol=args.tol, otreg_strength=args.otreg_strength),
            elif model_type == "attncnf":
                self.model = CombinedSpatiotemporalModel(SelfAttentiveCNF(**kwargs['model']), tpp_model).to(device)
                    # dim=x_dim, hidden_dims=list(map(int, args.hdims.split("-"))),
                    # layer_type=args.layer_type, actfn=args.actfn, l2_attn=args.l2_attn, tol=args.tol, otreg_strength=args.otreg_strength),
            else:
                raise ValueError(f"Invalid model {model_type}")

    def forward(self, event_times, event_types, mask):
        return self.model(event_times, event_types, mask, self.t_0, self.t_1)

    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        model.train()

        event, timestamps, mask = minibatch[0]

        space_loglik, time_loglik = model(
                timestamps, event, mask
        )
        space_sum = loss_f(space_loglik.sum())
        time_sum = loss_f(time_loglik.sum())
        loss = space_sum + time_sum
    
        loss.backward()
    
        time_loss = time_sum.item()
        space_loss = space_sum.item()
        fact = minibatch[1].sum()
    
        return time_loss, space_loss, fact
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()

        event, timestamps, mask = minibatch[0]

        space_loglik, time_loglik = model(
            timestamps, event, mask
        )
        space_sum = loss_f(space_loglik.sum())
        time_sum = loss_f(time_loglik.sum())
        loss = space_sum + time_sum
        
        loss = loss.item()
        time_loss = time_sum.item()
        space_loss = space_sum.item()
        fact = minibatch[1].sum()
    
        return time_loss, space_loss, fact
        
    def postprocess(input):
        '''
        [time absolute loss, spatio loss, time relative loss]
        '''
        return [input[0], input[1], input[0] - input[2]]

    def log_print_format(input):
        format_dict = {}
        format_dict['absolute_loss'] = input[0]
        format_dict['spatial_loss'] = input[1]
        format_dict['relative_loss'] = input[2]
        format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'spatial_loss': ':8.5f'}
        return format_dict

    format_dict_length = 3
    
    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f', 'spatial_loss': ':8.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['spatial_loss'] = input[1]
        format_dict['relative loss'] = input[2]
        return format_dict
    
    metric_number = 2 # metric number is the length of the output of choose_metric

    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [torch.abs(evaluation_report[-1]).item(), torch.abs(test_report[-1]).item()]
    

def loss_f(loglik):
    '''
    The definition of loss.
    '''
    return loglik.mul(-1.0)

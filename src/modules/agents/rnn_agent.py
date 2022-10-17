import torch.nn as nn
import torch.nn.functional as F
from modules.gnn_embedder import GIN
import torch

MAP_EMBEDDER = {
    'gin':      GIN
}


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.expansion_input = 1
        nagents = args.env_args['nagents']
        if hasattr(args, 'wifi_args') and args.wifi_args['comm_type'] != 'never':
            self.expansion_input = 1 + len(args.wifi_args['msg_type'])
            dim_hid = args.rnn_hidden_dim
            if 'hid' in args.wifi_args['msg_type']:
                self.msg_gradient = args.wifi_args['msg_gradient']
                name_arch = args.wifi_args['embedder_type']['hid']
                dim_msg = args.rnn_hidden_dim if not hasattr(args, 'dim_msg') else args.dim_msg
                self.embedder_comm = self._instantiate_embedder(name_arch, dim_msg, dim_hid, nagents)
            if 'rss' in args.wifi_args['msg_type']:
                name_arch = args.wifi_args['embedder_type']['rss']
                dim_msg = args.dim_wifi_measure
                self.embedder_measure = self._instantiate_embedder(name_arch, dim_msg, dim_hid, nagents)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim * self.expansion_input, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def _instantiate_embedder(self, name_arch, dim_msg, dim_hid, nagents):
        """
        Wrapper. We need this since mlp embedder needs an additional arg of nagents
        """
        embedder = MAP_EMBEDDER[name_arch]
        if name_arch == 'mlp':      # baseline, so need nagents
            return embedder(dim_msg, dim_hid, nagents)
        else:
            return embedder(dim_msg, dim_hid)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_(), None

    def forward(self, inputs, hidden_state, attention_keys):
        """
        Use heterogeneous GIN to embed the msg and rss info in the inputs. 
        """
        assert attention_keys is None, "rnn agent does not aggr msg with attention"
        if type(inputs) == dict:
            assert set(inputs.keys()).issubset(set(['inputs', 'comm_mask', 'comm_measure', 'comm_msg']))
            # TODO: try just use the generated 'hidden_state'. Don't use the stored comm_msg??
            #       but remember to detach
            assert len(hidden_state.shape) == 2 or len(hidden_state.shape) == 3
            inputs_orig = F.relu(self.fc1(inputs['inputs']))
            inputs_cat = [inputs_orig]
            if hasattr(self, 'embedder_comm'):
                if 'comm_msg' in inputs:
                    comm_msg = inputs['comm_msg']
                else:
                    _batch, _nagents = inputs['comm_mask'].shape[:2]
                    comm_msg = hidden_state.reshape(_batch, _nagents, -1)
                    if self.msg_gradient == 'detach':
                        comm_msg = comm_msg.clone().detach()
                inputs_comm = self.embedder_comm(comm_msg, inputs['comm_mask'])
                inputs_comm = inputs_comm.reshape(-1, inputs_comm.size(-1))
                inputs_cat.append(inputs_comm)
            if hasattr(self, 'embedder_measure'):
                inputs_measure = self.embedder_measure(inputs['comm_measure'], inputs['comm_mask'])
                inputs_measure = inputs_measure.reshape(-1, inputs_measure.size(-1))
                inputs_cat.append(inputs_measure)
            # concat inputs with embedded comm_mask & comm_measure
            x = torch.cat(inputs_cat, dim=-1)
        else:
            x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h, None

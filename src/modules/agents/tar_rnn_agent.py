import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from modules.agents.rnn_agent import RNNAgent, MAP_EMBEDDER


class TarRNNAgent(RNNAgent):
    def __init__(self, input_shape, args):
        super(TarRNNAgent, self).__init__(input_shape, args)
        if hasattr(args, 'wifi_args') and args.wifi_args['comm_type'] != 'never':
            dim_hid = args.rnn_hidden_dim
            name_arch = 'mlp-orig'
            if 'hid' in args.wifi_args['msg_type'] or 'rss' in args.wifi_args['msg_type']:
                self.nn_atten_key = self._instantiate_embedder(name_arch, dim_hid, dim_hid, 1)
                self.nn_atten_query = self._instantiate_embedder(name_arch, dim_hid, dim_hid, 1)

    def init_hidden(self):
        hid_init = self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        if hasattr(self, 'nn_atten_key'):
            attn_init = self.nn_atten_key(hid_init)
        return (
            hid_init,
            attn_init
        )

    def forward(self, inputs, hidden_state, attention_keys):
        ## TODO
        if type(inputs) == dict:
            assert set(inputs.keys()).issubset(set(['inputs', 'comm_mask', 'comm_measure', 'comm_msg', 'attn_msg']))
            assert len(hidden_state.shape) == 2 or len(hidden_state.shape) == 3
            if len(hidden_state.shape) == 2:       
                _batch, _nagents = inputs['comm_mask'].shape[:2]
                hidden_state = hidden_state.reshape(_batch, _nagents, -1)
            inputs_orig = F.relu(self.fc1(inputs['inputs']))
            inputs_cat = [inputs_orig]
            new_attn_key = self.nn_atten_key(hidden_state)
            cur_atten_query = self.nn_atten_key(hidden_state)
            assert hasattr(self, "embedder_comm")
            if 'comm_msg' in inputs:
                comm_msg = inputs['comm_msg']
                cur_atten_key = inputs['atten_msg']
            else:
                comm_msg = hidden_state
                cur_atten_key = attention_keys
                if self.msg_gradient == 'detach':
                    comm_msg = comm_msg.clone().detach()

            sim_key_query = (cur_atten_key[:, :, np.newaxis, :] * cur_atten_query[:, np.newaxis, :, :]).sum(dim=-1)
            w_attn = F.softmax(sim_key_query + (~inputs['comm_mask'] * -5000), dim=-1)
            inputs_mask = inputs['comm_mask'].float() * w_attn
            inputs_comm = self.embedder_comm(comm_msg, inputs_mask)
            inputs_comm = inputs_comm.reshape(-1, inputs_comm.size(-1))
            inputs_cat.append(inputs_comm)
            if hasattr(self, 'embedder_measure'):
                inputs_measure = self.embedder_measure(inputs['comm_measure'], inputs_mask)
                inputs_measure = inputs_measure.reshape(-1, inputs_measure.size(-1))
                inputs_cat.append(inputs_measure)
            x = torch.cat(inputs_cat, dim=-1)
        else:
            x = F.relu(self.fc1(inputs))
            new_attn_key = None
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h, new_attn_key

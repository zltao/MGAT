import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn.inits import uniform, glorot, zeros
class GraphGAT(MessagePassing):
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		super(GraphGAT, self).__init__(aggr=aggr, **kwargs)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.dropout = 0.2

		self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()
		self.is_get_attention = False

	def reset_parameters(self):
		uniform(self.in_channels, self.weight)
		uniform(self.in_channels, self.bias)


	def forward(self, x, edge_index, size=None):
		if size is None:
			edge_index, _ = remove_self_loops(edge_index)
		x = x.unsqueeze(-1) if x.dim() == 1 else x
		x = torch.matmul(x, self.weight)

		return self.propagate(edge_index, size=size, x=x)

	def message(self, edge_index_i, x_i, x_j, size_i, edge_index, size):
		# Compute attention coefficients.
		x_i = x_i.view(-1, self.out_channels)
		x_j = x_j.view(-1, self.out_channels)
		inner_product = torch.mul(x_i, F.leaky_relu(x_j)).sum(dim=-1)

		# gate
		row, col = edge_index
		deg = degree(row, size[0], dtype=x_i.dtype)
		deg_inv_sqrt = deg[row].pow(-0.5)
		tmp = torch.mul(deg_inv_sqrt, inner_product)
		gate_w = torch.sigmoid(tmp)
		# gate_w = F.dropout(gate_w, p=self.dropout, training=self.training)

		# attention
		tmp = torch.mul(inner_product, gate_w)
		attention_w = softmax(tmp, edge_index_i, size_i)
		#attention_w = F.dropout(attention_w, p=self.dropout, training=self.training)
		return torch.mul(x_j, attention_w.view(-1, 1))

	def update(self, aggr_out):
		if self.bias is not None:
			aggr_out = aggr_out + self.bias
		if self.normalize:
			aggr_out = F.normalize(aggr_out, p=2, dim=-1)
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

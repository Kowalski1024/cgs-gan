import torch
from torch import nn
import numpy as np
import math
from torch_utils.ops.neighbor_max import neighbor_max
from torch_utils.ops.geo_stats import geo_stats
from training.gaussian import trunc_exp
from training.topology import TopologyFactory



SCALE_MAX = 0.02
SCALE_MIN = 1e-6
SCALE_INIT = 0.01

def bounded_log_sigmoid(
    raw: torch.Tensor, log_min: float, log_max: float
) -> torch.Tensor:
    span = log_max - log_min
    return log_min + (span * torch.sigmoid(raw))


def fmm_modulate_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    activation: str = "demod",
) -> torch.Tensor:
    # x: [B, N, C], styles: [B, size]
    c_in = x.shape[-1]
    c_out, c_in_weight = weight.shape
    
    B = styles.shape[0]
    rank = styles.shape[1] // (c_in + c_out)
    assert styles.shape[1] % (c_in + c_out) == 0
    
    # Construct batched modulation: [B, c_out, c_in]
    left_matrix = styles[:, : c_out * rank].view(B, c_out, rank)  # [B, c_out, rank]
    right_matrix = styles[:, c_out * rank :].view(B, rank, c_in)  # [B, rank, c_in]
    modulation = torch.bmm(left_matrix, right_matrix) / np.sqrt(rank)  # [B, c_out, c_in]
    
    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5
    
    # Batched weight modulation: [B, c_out, c_in]
    W = weight.unsqueeze(0) * (modulation + 1.0)  # [B, c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8)  # [B, c_out, c_in]
    W = W.to(dtype=x.dtype)
    
    # Batched matmul: [B, N, c_in] @ [B, c_in, c_out] -> [B, N, c_out]
    out = torch.bmm(x, W.transpose(1, 2))

    return out


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        activation=nn.LeakyReLU(inplace=True),
        rank=10,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.affine = nn.Linear(self.w_dim, (in_channels + out_channels) * rank)

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x, w):
        styles = self.affine(w)

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod"
        )

        if self.bias is not None:
            x = x + self.bias.view(1, -1)  # Reshape bias for broadcasting

        noise = torch.randn(self.out_channels, device=x.device) * self.noise_strength
        x = self.activation(x + noise)

        return x


class GaussianEncoding(torch.nn.Module):
    """Fourier features like in f.py (cos/sin of random projections)."""

    def __init__(self, sigma: float, input_size: int, encoded_size: int):
        super().__init__()
        b = torch.randn((encoded_size, input_size)) * sigma
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vp = 2 * np.pi * x @ self.b.t()
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class IcoPointGNN(nn.Module):
    def __init__(self, channels, z_dim, **kwargs):
        super().__init__()
        
        # 1. Delta Predictor (h): x -> delta
        # We can optimize this by reducing the inner dimension if 'channels' is large
        self.mlp_h = nn.ModuleList([
            SynthesisLayer(channels, channels // 2, z_dim),
            SynthesisLayer(channels // 2, 3, z_dim, activation=nn.Tanh()),
        ])

        # 2. Update Network (g): (x + message) -> out
        # Input dim is channels (from max pool) + 3 (from max pool pos)
        self.mlp_g = nn.ModuleList([
            SynthesisLayer(channels + 3, channels, z_dim),
            SynthesisLayer(channels, channels, z_dim),
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        pos: [N, 3]
        edge_index: [N, 6]
        w: [L, B, z_dim]
        """
        
        # --- 1. Compute Delta (Alignment) ---
        # This part is strictly sequential
        delta = x
        for i, layer in enumerate(self.mlp_h):
            delta = layer(delta, w[:, 0])
        # --- 2. Feature Aggregation (Memory Optimized) ---
        # Gather features: [N, 6, C]
        # We immediately Max-Pool. We do NOT concatenate geometry yet.
        # This saves significant VRAM bandwidth.
        x_aggr, _ = x[:, edge_index].max(dim=2)  # [B, N, C]
        
        # --- 3. Geometric Aggregation ---
        # Formula: pos_j - (pos_i - delta_i)  ==> (pos_j - pos_i) + delta_i
        # pos[edge_index]: [N, 6, 3]
        # pos.unsqueeze(0).unsqueeze(2): [1, N, 1, 3]
        # delta.unsqueeze(2): [B, N, 1, 3]
        
        # We compute relative geometry and max-pool it immediately
        # (pos_j - pos_i) + delta
        pos_j = pos[edge_index].unsqueeze(0)  # [1, N, 6, 3]
        pos_i = pos.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 3]
        rel_pos = (pos_j - pos_i) + delta.unsqueeze(2)  # [B, N, 6, 3]
        pos_aggr, _ = rel_pos.max(dim=2) # [B, N, 3]
        
        # --- 4. Fusion ---
        # Now we concat small tensors [N, 3] and [N, C] -> [N, C+3]
        out = torch.cat([pos_aggr, x_aggr], dim=-1)
        
        # --- 5. Update State ---
        # Offset w index by number of layers in h
        w_offset = len(self.mlp_h)
        for i, layer in enumerate(self.mlp_g):
            out = layer(out, w[:, 0])
        # Residual connection
        return x + out


class IcoLINKX(nn.Module):
    """
    Optimized LINKX operator for fixed topology (Icosahedron).
    Replaces sparse adjacency matmul with dense embedding lookups.
    """
    def __init__(
        self,
        num_nodes: int,       # N (e.g. 10242 or 40962)
        in_channels: int,     # Dim of input x
        hidden_channels: int, # Internal dim
        out_channels: int,    # Output dim
        num_layers: int,      # Number of Synthesis Layers
        w_dim: int,           # Latent code dim
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels

        # 1. Edge/Structure Branch
        # Mathematically equivalent to SparseLinear(num_nodes, hidden)
        # We learn a weight vector W_j for every node j.
        # The operation is Sum_{j in neighbors} W_j
        self.edge_emb = nn.Embedding(num_nodes, hidden_channels)

        # 2. Node/Feature Branch
        # Input is x [B, N, C]
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(inplace=True),
        )

        # 3. Mixing Layers
        self.cat_lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = nn.Linear(hidden_channels, hidden_channels)

        # 4. Final Generative Layers (Modulated)
        # Note: Input to this block is hidden_channels
        gen_ch = [hidden_channels] * (num_layers + 1)
        # Fix the last dim to be out_channels
        gen_ch[-1] = out_channels
        
        self.final_mlp = nn.ModuleList()
        for i in range(num_layers):
            self.final_mlp.append(
                SynthesisLayer(gen_ch[i], gen_ch[i+1], w_dim)
            )

        self.final_act = nn.LeakyReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming init for embeddings to match SparseLinear behavior
        nn.init.kaiming_uniform_(self.edge_emb.weight, a=math.sqrt(5))
        # Reset other layers... (omitted for brevity, standard PyTorch init is usually fine)

    def forward(
        self,
        x: torch.Tensor,   # [B, N, in_C]
        dense_edge_index: torch.Tensor, # [N, 6]
        w: torch.Tensor,             # [L, B, W_dim] or list
    ) -> torch.Tensor:
        """
        dense_edge_index: Must handle padding for degree-5 nodes (e.g. repeat last neighbor).
        """
        B = x.shape[0]
        
        # --- Branch A: Structure (Edge) ---
        # 1. Retrieve neighbor weights: [N, 6] -> [N, 6, H]
        nb_weights = self.edge_emb(dense_edge_index)
        
        # 2. Aggregation (Sum): [N, 6, H] -> [N, H]
        # This computes A * W effectively
        out = nb_weights.sum(dim=1)
            
        # 4. First Mixing
        out = out + self.cat_lin1(out)
        
        # Expand structure features to batch size: [N, H] -> [B, N, H]
        out = out.unsqueeze(0).expand(B, -1, -1)

        # --- Branch B: Node Features ---
        # x: [B, N, C] -> [B, N, H]
        x_feat = self.node_mlp(x)
        
        # Combine
        out = out + x_feat
        out = out + self.cat_lin2(x_feat)

        # --- Branch C: Generative/Synthesis ---
        out = self.final_act(out)
        
        for i, layer in enumerate(self.final_mlp):
            # w[i] shape: [B, W_dim]
            out = layer(out, w[:, 0])
            
        return out


class Decoder(nn.Module):
    feature_channels = {"scales": 3, "rotation": 4, "opacity": 1, "color": 3, "xyz": 3}

    def __init__(self, in_channels=640, use_rgb=True, use_pc=True):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.decoders = torch.nn.ModuleList()

        for key, channels in self.feature_channels.items():
            layer = nn.Linear(256, channels)

            if key == "scales":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, np.log(0.1 / (1 - 0.1)))

            self.decoders.append(layer)

    def forward(self, x, pc=None):
        x = self.mlp(x)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v, p=2, dim=-1)
            elif k == "scales":
                v = trunc_exp(v)
                v = torch.clamp(v, min=0, max=0.02)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "color":
                v = torch.tanh(v) * 1.1
                v = (v + 1) / 2
            elif k == "xyz":
                # v = pc
                max_step = 1.2 / 32
                v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pc
            ret[k] = v

        return ret


class CloudGenerator(nn.Module):
    def __init__(self, channels=256, num_pts=1024, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim
        level = 5
        self.topology = TopologyFactory.precompute_icosahedron_stack(
            max_level=level,
            device='cpu',
        )[level]
        num_verts = self.topology.verts.shape[0]

        self.encoder = GaussianEncoding(
            sigma=10.0,
            input_size=3,
            encoded_size=64,
        )

        self.point_layers = nn.ModuleList()
        for _ in range(3):
            self.point_layers.append(
                IcoPointGNN(
                    channels=128,
                    z_dim=z_dim,
                )
            )

        self.attr_layers = nn.ModuleList()
        for _ in range(4):
            self.attr_layers.append(
                IcoLINKX(
                    num_nodes=num_verts,
                    in_channels=256,
                    hidden_channels=256,
                    out_channels=256,
                    num_layers=3,
                    w_dim=z_dim,
                )
            )

        self.attr_proj = SynthesisLayer(128, 256, z_dim)

        self.point_decoder = nn.ModuleList(
            [
                SynthesisLayer(128, 128, z_dim),
                SynthesisLayer(128, 64, z_dim),
                nn.Sequential(
                    nn.Linear(64, 3),
                    nn.Tanh(),
                )
            ]
        )

        self.attr_decoder = Decoder(in_channels=512, use_rgb=True, use_pc=True)

    def forward(self, w):
        B = w.shape[0]
        pos = self.topology.verts.to(w.device)  # [N, 3]
        edge_index = self.topology.dense_edge_index.to(w.device)  # [N, 6]
        x = self.encoder(pos)  # [N, C]
        x = x.unsqueeze(0).expand(B, -1, -1)  # [B, N, C]

        # Point Layers
        for layer in self.point_layers:
            x = layer(x, pos, edge_index, w)


        # global max_pool
        # x_pool, _ = x.max(dim=1)  # [B, C]
        # x_pool = self.global_conv(x_pool)  # [B, C]
        # x_pool_exp = x_pool.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, N, C]
        # x = torch.cat([x, x_pool_exp], dim=-1)  # [B, N, 2C]

        # Point Decoding
        new_pos = x
        for layer in self.point_decoder[:-1]:
            new_pos = layer(new_pos, w[:, 0])
        new_pos = self.point_decoder[-1](new_pos)  # [B, N, 3]

        # Attribute Layers
        x = self.attr_proj(x, w[:, 0])  # [B, N, 256]
        _x = x
        for layer in self.attr_layers:
            x = layer(x, edge_index, w)

        x = torch.cat([x, _x], dim=-1)  # [B, N, 512]

        # Attribute Decoding
        attrs = self.attr_decoder(x, pc=new_pos)

        xyz = attrs["xyz"]

        return (
            attrs["xyz"],
            attrs["scales"],
            attrs["rotation"],
            attrs["color"],
            attrs["opacity"],
        )
        

    def _apply(self, fn):
        """
        Override _apply to handle custom data structures.
        PyTorch calls this for .to(), .cuda(), .cpu(), .type(), etc.
        """
        super()._apply(fn)
        self.topology.map_tensors(fn)
        
        return self


class PointGenerator(nn.Module):
    def __init__(
        self,
        w_dim,
        options={},
    ):
        super().__init__()
        self.num_pts = options["num_pts"]
        self.point_encoder = CloudGenerator(num_pts=self.num_pts, z_dim=w_dim)
        self.point_encoder.sphere_pos_scale = options.get("init_pos_scale", 0.25)
        self.num_ws = 18
        self.z_dim = w_dim

    def forward(self, ws):
        return self.point_encoder(ws)
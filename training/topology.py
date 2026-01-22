import torch
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Callable

# Level 1: verts: [42, 3],      subdiv_map: [2, 30],      dense_edge_index: [42, 6]
# Level 2: verts: [162, 3],     subdiv_map: [2, 120],     dense_edge_index: [162, 6]
# Level 3: verts: [642, 3],     subdiv_map: [2, 480],     dense_edge_index: [642, 6]
# Level 4: verts: [2562, 3],    subdiv_map: [2, 1920],    dense_edge_index: [2562, 6]
# Level 5: verts: [10242, 3],   subdiv_map: [2, 7680],    dense_edge_index: [10242, 6]
# Level 6: verts: [40962, 3],   subdiv_map: [2, 30720],   dense_edge_index: [40962, 6]
# Level 7: verts: [163842, 3],  subdiv_map: [2, 122880],  dense_edge_index: [163842, 6]

@dataclass
class TopologyLevel:
    verts: torch.Tensor  # [N, 3] - Base positions
    subdiv_map: torch.Tensor  # [2, N_new] - Parents for subdivision
    dense_edge_index: torch.Tensor  # [N, 6] - The Dense Table for your Conv

    def map_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Applies a function 'fn' to all tensors in this level.
        Used by the parent module's .to(), .cuda(), .half() calls.
        """
        # fn handles device movement and type casting.
        # PyTorch's internal cast functions usually check is_floating_point(),
        # so it's safe to apply to indices, but explicit check is safer.

        if self.verts is not None:
            self.verts = fn(self.verts)

        # For indices, we typically only want device movement, not type casting (e.g. to fp16).
        # However, generic .to(device) is handled by fn.
        if self.subdiv_map is not None:
            self.subdiv_map = fn(self.subdiv_map)

        if self.dense_edge_index is not None:
            self.dense_edge_index = fn(self.dense_edge_index)
        return self


class TopologyStack(dict[int, TopologyLevel]):
    """Dict-like container for topology levels with helper tensor mapping.

    Behaves like a standard dict[int, TopologyLevel] (supports indexing,
    iteration, etc.) but adds map_tensors for convenient device/dtype moves.
    """

    def map_tensors(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        for lvl in self.values():
            lvl.map_tensors(fn)
        return self


class TopologyFactory:
    @staticmethod
    def get_edges_from_faces(faces):
        """Extract unique sorted edges."""
        edges = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        edges.sort(axis=1)
        return np.unique(edges, axis=0)

    @staticmethod
    def build_dense_neighbors(faces, num_verts, device: torch.device | str = "cpu"):
        """
        Constructs the [N, 6] neighbor table directly from faces.
        Pads degree-5 vertices by repeating the first neighbor.
        """
        # 1. Build Adjacency List
        # We use a list of sets first to ensure uniqueness, then convert to list
        adj = [set() for _ in range(num_verts)]

        for f in faces:
            v0, v1, v2 = f
            adj[v0].update([v1, v2])
            adj[v1].update([v0, v2])
            adj[v2].update([v0, v1])

        # 2. Convert to Fixed Tensor [N, 6]
        dense_table = np.zeros((num_verts, 6), dtype=np.int64)

        for i, neighbors in enumerate(adj):
            neigh_list = list(neighbors)
            k = len(neigh_list)

            if k == 6:
                dense_table[i, :] = neigh_list
            elif k == 5:
                # Pad degree 5 with the first neighbor (Circular padding)
                # This ensures the convolution doesn't read garbage memory
                dense_table[i, :5] = neigh_list
                dense_table[i, 5] = neigh_list[0]
            else:
                # Should not happen in Icosahedron subdivision (except maybe top/bottom poles)
                # Fallback: Pad with self or cycle
                # Create a cyclic iterator to fill up to 6
                fill = (neigh_list * 6)[:6]
                dense_table[i, :] = fill

        tensor = torch.from_numpy(dense_table)
        return tensor.to(torch.device(device))

    @staticmethod
    def subdivide_with_order(verts, faces):
        """Standard Loop Subdivision logic preserving vertex order."""
        edges_unique = TopologyFactory.get_edges_from_faces(faces)
        num_old_verts = len(verts)

        # Map edge (v1, v2) -> midpoint_idx
        edge_to_midpoint = {}
        for i, edge in enumerate(edges_unique):
            edge_to_midpoint[tuple(edge)] = num_old_verts + i

        new_faces = []
        for f in faces:
            v0, v1, v2 = f
            # Edges are sorted in get_edges, so we must query sorted
            m01 = edge_to_midpoint[tuple(sorted((v0, v1)))]
            m12 = edge_to_midpoint[tuple(sorted((v1, v2)))]
            m20 = edge_to_midpoint[tuple(sorted((v2, v0)))]

            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])

        new_faces = np.array(new_faces)

        # Calculate new positions (Spherical projection)
        old_pos = verts
        parents_a = verts[edges_unique[:, 0]]
        parents_b = verts[edges_unique[:, 1]]
        new_pos = (parents_a + parents_b) / 2.0
        # Normalize to keep on unit sphere
        new_pos = new_pos / np.linalg.norm(new_pos, axis=1, keepdims=True)

        full_verts = np.concatenate([old_pos, new_pos], axis=0)
        subdiv_map = edges_unique.T

        return full_verts, new_faces, subdiv_map

    @staticmethod
    def precompute_icosahedron_stack(
        start_level: int = 1,
        max_level: int = 7,
        device: torch.device | str = "cpu",
    ) -> TopologyStack:
        device = torch.device(device)

        # Start at Level 0
        mesh = trimesh.creation.icosahedron()
        curr_verts = mesh.vertices
        curr_faces = mesh.faces

        stack = {}

        # If user wants L0, we can add it, but usually we skip to L1
        # Loop to reach max_level
        for lvl in range(1, max_level + 1):
            # 1. Subdivide
            next_verts, next_faces, subdiv_map = TopologyFactory.subdivide_with_order(
                curr_verts, curr_faces
            )

            # 2. Store ONLY if we are >= start_level
            if lvl >= start_level:
                # Compute Dense Neighbors directly here
                neighbors = TopologyFactory.build_dense_neighbors(
                    next_faces, len(next_verts), device=device
                )

                stack[lvl] = TopologyLevel(
                    verts=torch.tensor(next_verts, dtype=torch.float32, device=device),
                    subdiv_map=torch.tensor(
                        subdiv_map, dtype=torch.long, device=device
                    ),
                    dense_edge_index=neighbors,
                )
                print(
                    f"  Level {lvl}: {len(next_verts)} verts. Neighbors shape: {neighbors.shape}"
                )

            curr_verts = next_verts
            curr_faces = next_faces

        return TopologyStack(stack)

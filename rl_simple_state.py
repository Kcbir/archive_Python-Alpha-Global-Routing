# coding: utf-8
"""
Simple Routing State — RustWorkX Backend
Fast graph state for VLSI global routing RL.

Key features:
  - RustWorkX graph (10x faster than NetworkX)
  - Vectorized feature computation (numpy)
  - PyG data export for GraphSAGE
  - Clean action/state interface for RL
"""

import numpy as np
import rustworkx as rx


class SimpleRoutingState:
    """
    Routing state backed by a RustWorkX PyGraph.

    Manages:
      * capacity matrix  (nLayers × ySize × xSize)
      * routing graph    (nodes = GCells, edges = legal moves)
      * feature extraction for GNN / CNN policies
    """

    ACTIONS = ['L', 'R', 'U', 'D', 'UP_LAYER', 'DOWN_LAYER']
    ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
    NUM_ACTIONS = len(ACTIONS)

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #
    def __init__(self, cap_matrix, layer_dirs):
        """
        Args:
            cap_matrix : np.ndarray (nLayers, ySize, xSize) – capacity values
            layer_dirs : list[int]  – 0 = horizontal (x), 1 = vertical (y)
        """
        self.cap = cap_matrix.copy().astype(np.float32)
        self.original_cap = cap_matrix.copy().astype(np.float32)
        self.layer_dirs = list(layer_dirs)
        self.nLayers, self.ySize, self.xSize = self.cap.shape
        self.graph = None
        self._build_graph()

    # ---- coordinate helpers ---- #
    def _c2i(self, z, y, x):
        """(z, y, x) → flat node index."""
        return z * self.ySize * self.xSize + y * self.xSize + x

    def _i2c(self, idx):
        """flat node index → (z, y, x)."""
        HW = self.ySize * self.xSize
        z = idx // HW
        r = idx % HW
        return (z, r // self.xSize, r % self.xSize)

    # ---- graph construction ---- #
    def _build_graph(self):
        """Build RustWorkX PyGraph from capacity matrix (vectorised)."""
        nL, H, W = self.nLayers, self.ySize, self.xSize
        self.graph = rx.PyGraph()
        self.graph.add_nodes_from(range(nL * H * W))

        edges = []

        # Routing edges (layer 0 is **not** routable — standard VLSI convention)
        for z in range(1, nL):
            base = z * H * W
            if self.layer_dirs[z] == 0:          # horizontal → x edges
                yy, xx = np.mgrid[:H, :W - 1]
                n1 = (base + yy * W + xx).ravel()
                n2 = n1 + 1
                w = (10.0
                     + (11.0 - self.cap[z, yy, xx]).ravel()
                     + (11.0 - self.cap[z, yy, xx + 1]).ravel())
                edges.extend(zip(n1.astype(int).tolist(),
                                 n2.astype(int).tolist(),
                                 w.astype(float).tolist()))

            elif self.layer_dirs[z] == 1:        # vertical → y edges
                yy, xx = np.mgrid[:H - 1, :W]
                n1 = (base + yy * W + xx).ravel()
                n2 = n1 + W
                w = (10.0
                     + (11.0 - self.cap[z, yy, xx]).ravel()
                     + (11.0 - self.cap[z, yy + 1, xx]).ravel())
                edges.extend(zip(n1.astype(int).tolist(),
                                 n2.astype(int).tolist(),
                                 w.astype(float).tolist()))

        # Via edges (connect adjacent layers)
        for z in range(nL - 1):
            b1 = z * H * W
            b2 = (z + 1) * H * W
            idx = np.arange(H * W)
            edges.extend(zip((b1 + idx).tolist(),
                             (b2 + idx).tolist(),
                             [40.0] * len(idx)))

        if edges:
            self.graph.add_edges_from(edges)

    # ------------------------------------------------------------------ #
    #  RL interface                                                        #
    # ------------------------------------------------------------------ #
    def get_valid_actions(self, coord):
        """
        Return dict  action_name → next_coord  for *coord* = (z, y, x).
        Respects layer direction and grid boundaries.
        """
        z, y, x = coord
        valid = {}
        # Horizontal moves (only on horizontal layers, skip layer 0)
        if 1 <= z < self.nLayers and self.layer_dirs[z] == 0:
            if x > 0:
                valid['L'] = (z, y, x - 1)
            if x < self.xSize - 1:
                valid['R'] = (z, y, x + 1)
        # Vertical moves (only on vertical layers, skip layer 0)
        if 1 <= z < self.nLayers and self.layer_dirs[z] == 1:
            if y > 0:
                valid['U'] = (z, y - 1, x)
            if y < self.ySize - 1:
                valid['D'] = (z, y + 1, x)
        # Layer changes (vias)
        if z < self.nLayers - 1:
            valid['UP_LAYER'] = (z + 1, y, x)
        if z > 0:
            valid['DOWN_LAYER'] = (z - 1, y, x)
        return valid

    def apply_action(self, coord, action):
        """Apply *action* at *coord*.  Returns new coordinate or None."""
        return self.get_valid_actions(coord).get(action)

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #
    def to_numpy_features(self, source, target):
        """
        Full-grid feature tensor (for CNN or small-grid training).

        Returns
        -------
        np.ndarray  shape (5, nLayers, ySize, xSize)
            channels: capacity, utilisation, dist-to-source, dist-to-target,
                      layer-direction
        """
        nL, H, W = self.nLayers, self.ySize, self.xSize
        mc = float(self.original_cap.max()) + 1e-6
        md = float(nL + H + W)

        f_cap  = self.cap / mc
        f_util = np.clip(1.0 - self.cap / (self.original_cap + 1e-6), 0, 1)

        zz, yy, xx = np.mgrid[:nL, :H, :W]
        f_ds = (np.abs(zz - source[0])
                + np.abs(yy - source[1])
                + np.abs(xx - source[2])).astype(np.float32) / md
        f_dt = (np.abs(zz - target[0])
                + np.abs(yy - target[1])
                + np.abs(xx - target[2])).astype(np.float32) / md

        f_ld = np.zeros((nL, H, W), dtype=np.float32)
        for z in range(min(nL, len(self.layer_dirs))):
            f_ld[z] = float(self.layer_dirs[z])

        return np.stack([f_cap, f_util, f_ds, f_dt, f_ld], axis=0)

    def to_pyg_data(self, source, target):
        """
        Convert state to a PyTorch-Geometric ``Data`` object for GraphSAGE.

        Node features (8-dim):
            capacity, utilisation, dist-source, dist-target,
            layer-dir, z-norm, is-source, is-target

        Returns ``None`` when torch-geometric is not installed.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            return None

        nL, H, W = self.nLayers, self.ySize, self.xSize
        N = nL * H * W
        mc = float(self.original_cap.max()) + 1e-6
        md = float(nL + H + W)

        zz, yy, xx = np.mgrid[:nL, :H, :W]
        zf = zz.ravel().astype(np.float32)
        yf = yy.ravel().astype(np.float32)
        xf = xx.ravel().astype(np.float32)
        cf = self.cap.ravel()
        of = self.original_cap.ravel()

        ld_arr = np.array(
            self.layer_dirs + [0] * max(0, nL - len(self.layer_dirs)),
            dtype=np.float32,
        )

        feat = np.column_stack([
            cf / mc,                                                          # 0 capacity
            np.clip(1.0 - cf / (of + 1e-6), 0, 1),                          # 1 utilisation
            (np.abs(zf - source[0]) + np.abs(yf - source[1])
             + np.abs(xf - source[2])) / md,                                 # 2 dist-src
            (np.abs(zf - target[0]) + np.abs(yf - target[1])
             + np.abs(xf - target[2])) / md,                                 # 3 dist-tgt
            ld_arr[zz.ravel().astype(int)],                                   # 4 layer-dir
            zf / max(nL - 1, 1),                                              # 5 z-norm
            np.eye(N, dtype=np.float32)[self._c2i(*source)]
                if self._c2i(*source) < N else np.zeros(N, np.float32),       # 6 is-source
            np.eye(N, dtype=np.float32)[self._c2i(*target)]
                if self._c2i(*target) < N else np.zeros(N, np.float32),       # 7 is-target
        ])

        x = torch.tensor(feat, dtype=torch.float32)

        el = self.graph.edge_list()
        if el:
            s = [e[0] for e in el]
            d = [e[1] for e in el]
            edge_index = torch.tensor([s + d, d + s], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    # ------------------------------------------------------------------ #
    #  State updates                                                       #
    # ------------------------------------------------------------------ #
    def update_after_routing(self, path):
        """Decrease capacity by 1 along *path* (list of (z, y, x))."""
        for z, y, x in path:
            self.cap[z, y, x] = max(0, self.cap[z, y, x] - 1.0)

    def reset_capacity(self):
        """Restore capacity to original values."""
        self.cap[:] = self.original_cap

    def get_overflow(self):
        """Total overflow (sum of negative capacities)."""
        return float(np.maximum(-self.cap, 0).sum())

    def summary(self):
        """Quick stats dict."""
        total = float(self.original_cap.sum())
        used = total - float(self.cap.sum())
        return {
            'nodes': self.graph.num_nodes() if self.graph else 0,
            'edges': self.graph.num_edges() if self.graph else 0,
            'grid': f'{self.nLayers}x{self.ySize}x{self.xSize}',
            'utilization': used / (total + 1e-6),
            'overflow': self.get_overflow(),
        }

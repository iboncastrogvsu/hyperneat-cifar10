import numpy as np

def grid_positions(width, height, channels=1):
    """
    Generate a 3D grid of neuron positions for an input substrate.

    Args:
        width (int): Number of x-coordinates (image width).
        height (int): Number of y-coordinates (image height).
        channels (int): Number of channels (e.g., RGB).

    Returns:
        np.ndarray: Array of shape (width*height*channels, 3), where each row is (x, y, z).
    """
    xs = np.linspace(-1.0, 1.0, width)
    ys = np.linspace(-1.0, 1.0, height)
    pos = []
    for c in range(channels):
        # Spread channels along the z-axis
        z = -1.0 + 2.0 * (c / max(1, channels-1)) if channels > 1 else 0.0
        for yi in ys:
            for xi in xs:
                pos.append((xi, yi, z))
    return np.array(pos, dtype=float)

def output_positions(num_outputs=10):
    """
    Define positions for output neurons spaced along the x-axis.

    Args:
        num_outputs (int): Number of output neurons.

    Returns:
        np.ndarray: Array of shape (num_outputs, 3).
    """
    xs = np.linspace(-1.0, 1.0, num_outputs)
    pos = [(x, 0.0, 1.0) for x in xs]
    return np.array(pos, dtype=float)

def pair_features(pre_pos, post_pos):
    """
    Compute pairwise features between pre- and post-synaptic neuron positions.

    Features include:
        - Pre-synaptic (x, y)
        - Post-synaptic (x, y)
        - Euclidean distance between positions
        - Bias term (constant 1.0)

    Args:
        pre_pos (np.ndarray): Positions of pre-synaptic neurons (N, 3).
        post_pos (np.ndarray): Positions of post-synaptic neurons (M, 3).

    Returns:
        np.ndarray: Feature matrix of shape (N*M, 6).
    """
    pre = pre_pos[:, :2]
    post = post_pos[:, :2]

    # Repeat and tile to form all possible pre-post pairs
    pre_repeat = np.repeat(pre, post.shape[0], axis=0)
    post_tile = np.tile(post, (pre.shape[0], 1))

    # Euclidean distance between positions
    d = np.linalg.norm(pre_repeat - post_tile, axis=1, keepdims=True)

    # Bias term
    bias = np.ones((pre_repeat.shape[0], 1), dtype=float)

    # Concatenate all features
    feats = np.concatenate([pre_repeat, post_tile, d, bias], axis=1)
    return feats

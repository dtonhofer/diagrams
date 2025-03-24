from typing import Final, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as mplaxes

# Draw "sliding average" graphs of a few noisy functions

# https://numpy.org/doc/stable/reference/generated/numpy.arange.html
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
# https://numpy.org/doc/stable/reference/generated/numpy.sin.html
# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace
# https://numpy.org/doc/stable/reference/generated/numpy.multiply.html


def build_bias_corr(size: int, beta: float) -> np.ndarray:
    bias_corr = np.zeros((size, 1), dtype=np.float64)
    # what is the bias correction for the first value? set it to 1
    bias_corr[0, 0] = 1.0
    for i in range(1, size):
        bias_corr[i, 0] = 1.0 / (1.0 - beta ** i)
    return bias_corr


def build_v_zeros_if_negative(theta: np.ndarray, beta: float) -> np.ndarray:
    size: Final[int] = theta.shape[0]
    v_zeros_if_negative = np.zeros((size, 1), dtype=np.float64)
    v_zeros_if_negative[0, 0] = (1 - beta) * theta[0, 0]
    for i in range(1, size):
        v_zeros_if_negative[i, 0] = (1 - beta) * theta[i, 0] + beta * v_zeros_if_negative[i - 1, 0]
    return v_zeros_if_negative


def build_v_starts_with_theta0(theta: np.ndarray, beta: float) -> np.ndarray:
    size: Final[int] = theta.shape[0]
    v_starts_with_theta0 = np.zeros((size, 1), dtype=np.float64)
    v_starts_with_theta0[0, 0] = theta[0, 0]
    for i in range(1, size):
        v_starts_with_theta0[i, 0] = (1 - beta) * theta[i, 0] + beta * v_starts_with_theta0[i - 1, 0]
    return v_starts_with_theta0


def build_v_with_bias_corr(theta: np.ndarray, bias_corr: np.ndarray, beta: float) -> np.ndarray:
    return np.multiply(build_v_zeros_if_negative(theta, beta), bias_corr)


def build_v_sliding_avg(theta: np.ndarray, beta: float) -> Tuple[np.ndarray, int]:
    size: Final[int] = theta.shape[0]
    avg_size = np.rint(1.0 / (1.0 - beta)).astype(int)
    v_sliding_avg = np.zeros((size, 1), dtype=np.float64)
    for i in range(avg_size - 1, size):
        v_sliding_avg[i, 0] = np.sum(theta[i - avg_size + 1:i, 0], axis=0) / avg_size
    return (v_sliding_avg, avg_size)


def build_v(theta: np.ndarray, bias_corr: np.ndarray, beta: float) -> Dict:
    v_zeros_if_negative = build_v_zeros_if_negative(theta, beta)
    v_starts_with_theta0 = build_v_starts_with_theta0(theta, beta)
    v_bias_corr = build_v_with_bias_corr(theta, bias_corr, beta)
    v_sliding_avg, avg_size = build_v_sliding_avg(theta, beta)
    max_y = np.max([np.max(theta),
                    np.max(v_zeros_if_negative),
                    np.max(v_starts_with_theta0),
                    np.max(v_bias_corr),
                    np.max(v_sliding_avg)])
    min_y = np.min([np.min(theta),
                    np.min(v_zeros_if_negative),
                    np.min(v_starts_with_theta0),
                    np.min(v_bias_corr),
                    np.min(v_sliding_avg)])
    return {"theta": theta,
            "v_zeros_if_negative": v_zeros_if_negative,
            "v_starts_with_theta0": v_starts_with_theta0,
            "v_bias_corr": v_bias_corr,
            "v_sliding_avg": v_sliding_avg,
            "avg_size": avg_size,
            "y_range": (min_y, max_y)}


def find_limits(x: np.ndarray, dict_list: List[Dict]) -> Dict:
    min_x = x[0, 0]
    max_x = x[-1, 0]
    width = max_x - min_x
    min_y_list = []
    max_y_list = []
    for dict in dict_list:
        min_y, max_y = dict["y_range"]
        min_y_list.append(min_y)
        max_y_list.append(max_y)
    min_y = min(min_y_list)
    max_y = max(max_y_list)
    height = max_y - min_y
    return {"min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "width": width,
            "height": height}


def set_axis_limits(ax: mplaxes.Axes, limits: Dict) -> None:
    min_x = limits["min_x"]
    max_x = limits["max_x"]
    min_y = limits["min_y"]
    max_y = limits["max_y"]
    width = limits["width"]
    height = limits["height"]
    ax.set_xlim(min_x - width / 10, max_x + width / 10)
    ax.set_ylim(min_y - height / 10, max_y + height / 10)
    ax.set_aspect(aspect='equal')


def plot_sublot_noisy(x: np.ndarray, ax: mplaxes.Axes, dict_noisy_foo: Dict, noise_std_deviation, name: str) -> None:
    avg_size = dict_noisy_foo["avg_size"]
    ax.plot(x, dict_noisy_foo["theta"], label=f'{name} (σ = {noise_std_deviation:.2f})', marker='x', linestyle='none')
    ax.plot(x, dict_noisy_foo["v_zeros_if_negative"], label='v, v(x<0) = 0', marker='.', linestyle='none')
    ax.plot(x, dict_noisy_foo["v_starts_with_theta0"], label='v, v(0) = Θ(0)', marker='.', linestyle='none')
    ax.plot(x, dict_noisy_foo["v_bias_corr"], label='v, v(x<0) = 0, bias corr.', marker='.', linestyle='none')
    restricted_x = x[avg_size - 1:, 0]
    restricted_v = dict_noisy_foo["v_sliding_avg"][avg_size - 1:, 0]
    ax.plot(restricted_x, restricted_v, label=f'v, sliding avg (window size = {avg_size})', marker='.', linestyle='none')
    ax.legend()
    ax.grid()


def plot_sublot_nonoise(x: np.ndarray, ax: mplaxes.Axes, dict_noisy_foo: Dict, name: str) -> None:
    avg_size = dict_noisy_foo["avg_size"]
    ax.plot(x, dict_noisy_foo["theta"], label=f'{name}', marker='x', linestyle='none')
    ax.plot(x, dict_noisy_foo["v_zeros_if_negative"], label='v, v(x<0) = 0', marker='.', linestyle='none')
    ax.plot(x, dict_noisy_foo["v_starts_with_theta0"], label='v, v(0) = Θ(0)', marker='.', linestyle='none')
    ax.plot(x, dict_noisy_foo["v_bias_corr"], label='v, v(x<0) = 0, bias corr.', marker='.', linestyle='none')
    restricted_x = x[avg_size - 1:, 0]
    restricted_v = dict_noisy_foo["v_sliding_avg"][avg_size - 1:, 0]
    ax.plot(restricted_x, restricted_v, label=f'v, sliding avg (window size = {avg_size})', marker='.', linestyle='none')
    ax.legend()
    ax.grid()


def plot(x: np.ndarray,
         dict_noisy_line: Dict,
         dict_noisy_sinus: Dict,
         dict_swinging: Dict,
         dict_cosine: Dict,
         noise_std_deviation) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # "axes" is Axes or array of Axes https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes
    ax_noisy_line = axes[0, 0]
    ax_noisy_sinus = axes[0, 1]
    ax_swinging = axes[1, 0]
    ax_cosine = axes[1, 1]
    limits: Dict = find_limits(x, [dict_noisy_line, dict_noisy_sinus, dict_swinging, dict_cosine])
    set_axis_limits(ax_noisy_line, limits)
    set_axis_limits(ax_noisy_sinus, limits)
    set_axis_limits(ax_swinging, limits)
    set_axis_limits(ax_cosine, limits)

    plot_sublot_noisy(x, ax_noisy_line, dict_noisy_line, noise_std_deviation, "noisy line")
    plot_sublot_noisy(x, ax_noisy_sinus, dict_noisy_sinus, noise_std_deviation, "noisy sinus")
    plot_sublot_nonoise(x, ax_swinging, dict_swinging, "superposed sin,cos,sin")
    plot_sublot_nonoise(x, ax_cosine, dict_cosine, "cosine")
    plt.tight_layout()
    plt.show()


def build_noisy_sinus(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
    two_pi: Final[float] = 2 * np.pi
    size = x.shape[0]
    period_num: Final[float] = 2  # number of periods of noisy sinus
    amplitude: Final[float] = 10  # amplitude of noisy sinus
    y_offset: Final[float] = 10  # offset of noisy sinus
    x_values: Final[np.ndarray] = np.linspace(0.0, two_pi * period_num, num=size, dtype=np.float64).reshape(-1, 1)
    assert x_values[0, 0] == 0.0
    assert np.isclose(x_values[size - 1, 0].squeeze(), two_pi * period_num)
    return np.sin(x_values) * amplitude + noise + y_offset


def build_swinging(x: np.ndarray) -> np.ndarray:
    two_pi: Final[float] = 2 * np.pi
    size = x.shape[0]
    x1_values: Final[np.ndarray] = np.linspace(0.0, two_pi * 0.5, num=size, dtype=np.float64).reshape(-1, 1)
    x2_values: Final[np.ndarray] = np.linspace(0.0, two_pi * 2, num=size, dtype=np.float64).reshape(-1, 1)
    x3_values: Final[np.ndarray] = np.linspace(0.0, two_pi * 4, num=size, dtype=np.float64).reshape(-1, 1)
    return np.sin(x1_values) * 20 + np.cos(x2_values) * 3 + np.sin(x3_values)


def build_cosine(x: np.ndarray) -> np.ndarray:
    two_pi: Final[float] = 2 * np.pi
    size = x.shape[0]
    x_values: Final[np.ndarray] = np.linspace(0.0, two_pi * 2, num=size, dtype=np.float64).reshape(-1, 1)
    return np.cos(x_values) * 10


def build_noisy_line(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
    slope: Final[float] = 0.5  # slope of noisy line
    return x * slope + noise


def main():
    # use this initialized RNG for reproducible results:
    # rng = np.random.default_rng(seed=42)
    # use this RNG that depends on current time for interesting results:
    rng = np.random.default_rng()
    #
    beta: Final[float] = 0.9  # beta of the running averages (factor applied to the previous average)
    size: Final[int] = 50  # number of data points in the graph
    #
    x: Final[np.ndarray] = np.arange(0, size, dtype=np.int32).reshape(-1, 1)
    assert x.shape == (size, 1)
    #
    noise_std_deviation: Final[float] = 4.0  # stddev of normal noise
    noise: Final[np.ndarray] = rng.normal(size=size, scale=noise_std_deviation).reshape(-1, 1)
    assert noise.shape == (size, 1)
    #
    bias_corr: Final[np.ndarray] = build_bias_corr(size, beta)
    #
    noisy_line: Final[np.ndarray] = build_noisy_line(x, noise)
    noisy_sinus: Final[np.ndarray] = build_noisy_sinus(x, noise)
    swinging: Final[np.ndarray] = build_swinging(x)
    cosine: Final[np.ndarray] = build_cosine(x)
    #
    dict_noisy_line = build_v(noisy_line, bias_corr, beta)
    dict_noisy_sinus = build_v(noisy_sinus, bias_corr, beta)
    dict_swinging = build_v(swinging, bias_corr, beta)
    dict_cosine = build_v(cosine, bias_corr, beta)
    #
    plot(x, dict_noisy_line, dict_noisy_sinus, dict_swinging, dict_cosine, noise_std_deviation)


main()

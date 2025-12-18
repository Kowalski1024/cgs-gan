import PIL.Image
import numpy as np
import torch
import cv2


def setup_snapshot_image_grid(training_set, random_seed=0, gw=8, gh=8):
    rnd = np.random.RandomState(random_seed)
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


def save_image_grid(img, fname, drange, grid_size, wandb_logger=None):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    if wandb_logger is not None:
        import wandb
        log_img = wandb.Image(img)
        wandb_logger.log({"samples": log_img})
    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def _images_to_spectrum(img: np.ndarray) -> np.ndarray:
    """Convert a batch of images (NCHW) to log-magnitude FFT spectra (N1HW).

    Expects images in any numeric range; normalization happens internally.
    """
    img = np.asarray(img)
    assert img.ndim == 4, f"Expected NCHW array, got shape {img.shape}"
    n, c, h, w = img.shape
    assert c in (1, 3), f"Expected C in (1, 3), got {c}"

    # Convert to grayscale for a single, readable spectrum image.
    x = img.astype(np.float32)
    if c == 3:
        # Simple luminance; input ordering is assumed RGB.
        x = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
    else:
        x = x[:, 0]

    # FFT -> log-magnitude spectrum.
    fft = np.fft.fft2(x, axes=(-2, -1))
    fft = np.fft.fftshift(fft, axes=(-2, -1))
    mag = np.abs(fft)
    spec = np.log1p(mag)

    # Robust per-image normalization to [0, 255].
    spec_out = np.empty((n, 1, h, w), dtype=np.uint8)
    for i in range(n):
        s = spec[i]
        # Clip bright outliers so the spectrum is visible.
        hi = np.percentile(s, 99.5)
        lo = float(s.min())
        hi = float(hi)
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        s = np.clip(s, lo, hi)
        s = (s - lo) * (255.0 / (hi - lo))
        spec_out[i, 0] = np.rint(s).clip(0, 255).astype(np.uint8)
    return spec_out


def save_spectrum_image_grid(img, fname, grid_size, wandb_logger=None):
    """Save a grid of FFT spectrum-domain images.

    Input format matches save_image_grid(): img is NCHW.
    """
    spec = _images_to_spectrum(img)
    gw, gh = grid_size
    _n, c, h, w = spec.shape
    spec = spec.reshape([gh, gw, c, h, w]).transpose(0, 3, 1, 4, 2).reshape([gh * h, gw * w, c])

    if wandb_logger is not None:
        import wandb

        wandb_logger.log({"samples_spectrum": wandb.Image(spec[:, :, 0])})

    PIL.Image.fromarray(spec[:, :, 0], 'L').save(fname)




def azimuthal_average(image: np.ndarray, center=None) -> np.ndarray:
    """Azimuthally averaged radial profile (bin size = 1 pixel).

    Matches the commonly used implementation from astrobetter.
    """
    image = np.asarray(image)
    assert image.ndim == 2, f"Expected 2D image, got shape {image.shape}"

    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])
    r_int = r.astype(np.int64)
    rmax = int(r_int.max())

    # Bin by integer radius.
    r_flat = r_int.reshape(-1)
    i_flat = image.reshape(-1).astype(np.float64)
    sums = np.bincount(r_flat, weights=i_flat, minlength=rmax + 1).astype(np.float64)
    counts = np.bincount(r_flat, minlength=rmax + 1).astype(np.float64)
    counts[counts == 0] = 1.0
    return (sums / counts).astype(np.float32)


def radial_power_spectrum(
    img: np.ndarray,
    max_freq: float = 0.5,
    epsilon: float = 1e-8,
    per_image_normalize: bool = True,
):
    """Compute mean/std 1D power spectrum like the provided reference snippet.

    Steps per image:
      1) grayscale
      2) FFT2 + shift
      3) magnitude spectrum: 20 * log(|shifted_fft| + epsilon)
      4) azimuthal average (integer-radius bins)
      5) min-max normalize to [0, 1] (optional)

    Assumes square images; if not square, center-crops to a square.
    """
    img = np.asarray(img)
    assert img.ndim == 4, f"Expected NCHW array, got shape {img.shape}"
    n, c, h, w = img.shape
    assert c in (1, 3), f"Expected C in (1, 3), got {c}"

    # Convert to grayscale.
    x = img.astype(np.float32)
    if c == 3:
        x = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
    else:
        x = x[:, 0]

    # Enforce square assumption by center-cropping if needed.
    if h != w:
        m = min(h, w)
        y0 = (h - m) // 2
        x0 = (w - m) // 2
        x = x[:, y0 : y0 + m, x0 : x0 + m]
        h = w = m

    max_freq = float(max_freq)
    if not (0.0 < max_freq <= 0.5):
        raise ValueError("max_freq must be in (0, 0.5]")
    kmax = h // 2
    freqs_full = (np.arange(kmax, dtype=np.float32) / float(h))
    keep_k = freqs_full <= max_freq

    per_image = np.empty((n, kmax), dtype=np.float32)
    for t in range(n):
        fft = np.fft.fft2(x[t])
        fshift = np.fft.fftshift(fft)
        magnitude_spectrum = 20.0 * np.log(np.abs(fshift) + float(epsilon))
        psd1d = azimuthal_average(magnitude_spectrum)
        psd1d = psd1d[:kmax]
        if per_image_normalize:
            lo = float(psd1d.min())
            hi = float(psd1d.max())
            if not np.isfinite(hi) or hi <= lo:
                psd1d = np.zeros_like(psd1d)
            else:
                psd1d = (psd1d - lo) / (hi - lo)
        per_image[t] = psd1d

    per_image = per_image[:, keep_k]
    freqs = freqs_full[keep_k]
    mean = per_image.mean(axis=0)
    std = per_image.std(axis=0)
    return freqs, mean, std


def save_radial_power_spectrum_plot(img, fname, wandb_logger=None, num_bins: int | None = None, max_freq: float = 0.5):
    """Save a line plot of spatial frequency vs power spectrum (mean and std).

    Plots mean with a transparent +/- std band.
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"WARNING: matplotlib unavailable; skipping power spectrum plot: {e}")
        return

    # Reference-style: magnitude spectrum -> azimuthal average -> per-image min-max normalization.
    freqs, mean, std = radial_power_spectrum(img, max_freq=max_freq)
    mean_y = mean
    lo_y = np.clip(mean - std, 0.0, 1.0)
    hi_y = np.clip(mean + std, 0.0, 1.0)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freqs, mean_y, label='mean')
    ax.fill_between(freqs, lo_y, hi_y, alpha=0.25, label='±1 std')
    ax.set_xlabel('Spatial Frequency (cycles/pixel)')
    ax.set_ylabel('Power Spectrum (normalized)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

    if wandb_logger is not None:
        import wandb

        wandb_logger.log({"power_spectrum": wandb.Image(fname)})


def print_stats(_dict):
    for k in _dict.keys():
        if k == "_features_rest":
            continue
        print("{} shape: {}, min: {} max: {}".format(k, _dict[k].shape, _dict[k].min(), _dict[k].max()))


def save_images(rgb_image, depth_image, device=None):
    if str(device) == "cuda:0":
        # save intermediate image for debugging
        temp_img = rgb_image.detach().cpu().numpy()[0]
        temp_img = np.clip(((temp_img + 1) * 127.5), 0, 255)
        cv2.imwrite(
            "temp_saves/temp_G_{}_{}.jpg".format(str(rgb_image.device), 0),
            temp_img.transpose([1, 2, 0])[:, :, ::-1],
        )
        temp_img = rgb_image.detach().cpu().numpy()[1]
        temp_img = np.clip(((temp_img + 1) * 127.5), 0, 255)
        cv2.imwrite(
            "temp_saves/temp_G_{}_{}.jpg".format(str(rgb_image.device), 1),
            temp_img.transpose([1, 2, 0])[:, :, ::-1],
        )

        temp_depth = depth_image.detach().cpu().numpy()[0]
        temp_depth = (temp_depth - temp_depth.min()) / (temp_depth.max() - temp_depth.min()) * 255
        cv2.imwrite(
            "temp_saves/depth_G_{}_{}.jpg".format(str(depth_image.device), 0),
            temp_depth.transpose([1, 2, 0])[:, :, ::-1],
        )


def slerp(v0, v1, t):
    v0 = torch.nn.functional.normalize(v0, p=2, dim=-1)
    v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)

    dot_product = torch.einsum("bi,bi->b", v0, v1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = torch.acos(dot_product)

    sin_theta = torch.sin(theta)
    interpolated_vector = (torch.sin((1 - t) * theta) / sin_theta)[:, None] * v0 + (torch.sin(t * theta) / sin_theta)[
        :, None
    ] * v1

    return interpolated_vector

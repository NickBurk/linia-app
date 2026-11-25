"""Directional terrain responses, multiscale components and ridge extraction."""

from dataclasses import dataclass
from math import ceil, pi

import numpy as np
from scipy import fft, ndimage

from lineament_analyzer.algorithms.results import build_path_result
from lineament_analyzer.algorithms.support import detector_support_mask
from lineament_analyzer.algorithms.tracing import TraceParameters, _trace
from lineament_analyzer.core.windowed import check_cancel


@dataclass(frozen=True)
class FilterParameters:
    levels: int = 3
    threshold: float = 0.28
    minimum_length_px: int = 12
    max_lines: int = 48


@dataclass(frozen=True)
class DirectionalParameters(FilterParameters):
    angle_start_deg: float = 0
    angle_end_deg: float = 180
    angle_step_deg: float = 22.5
    first_scale_px: float = 1.0


@dataclass(frozen=True)
class LogGaborParameters(DirectionalParameters):
    first_frequency: float = 0.25
    bandwidth: float = 0.55


@dataclass(frozen=True)
class WaveletParameters(FilterParameters):
    pass


@dataclass(frozen=True)
class ShearletParameters(FilterParameters):
    pass


@dataclass(frozen=True)
class Response:
    strength: np.ndarray
    tangent_rad: np.ndarray
    valid: np.ndarray


def _filled(dem):
    valid = dem.valid & np.isfinite(dem.values)
    if valid.all():
        return dem.values, valid
    if not valid.any():
        return np.zeros(valid.shape), valid
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return dem.values[tuple(nearest)], valid


def _angles(p):
    if not (
        0 <= p.angle_start_deg < p.angle_end_deg <= 180 and 5 <= p.angle_step_deg <= 90
        and 0.6 <= p.first_scale_px <= 8
    ):
        raise ValueError("Invalid directional filter angle range, step or scale")
    return np.deg2rad(np.arange(p.angle_start_deg, p.angle_end_deg, p.angle_step_deg))


def directional_components(image, p, cancel=None):
    """Scale-normalized Hessian responses along normals to the selected tangents."""
    angles = _angles(p)
    for level in range(p.levels):
        check_cancel(cancel)
        sigma = p.first_scale_px * 2**level
        xx = ndimage.gaussian_filter(image, sigma, order=(0, 2))
        xy = ndimage.gaussian_filter(image, sigma, order=(1, 1))
        yy = ndimage.gaussian_filter(image, sigma, order=(2, 0))
        # Remove residual DC from truncated sampled Gaussian derivative kernels.
        smooth = ndimage.gaussian_filter(image, sigma)
        impulse = np.zeros(2 * round(4 * sigma) + 1)
        impulse[len(impulse) // 2] = 1
        dc = ndimage.gaussian_filter1d(impulse, sigma, order=2, mode="constant").sum()
        xx -= dc * smooth
        yy -= dc * smooth
        for angle in angles:
            check_cancel(cancel)
            nx, ny = -np.sin(angle), np.cos(angle)
            response = nx**2 * xx + 2 * nx * ny * xy + ny**2 * yy
            yield angle, sigma**2 * response


def wavelet_components(image, levels, cancel=None):
    """Undecimated separable B-spline wavelet decomposition with residual."""
    if not 1 <= levels <= 5:
        raise ValueError("Wavelet levels must be in [1, 5]")
    smooth = np.asarray(image, dtype=float)
    for level in range(levels):
        check_cancel(cancel)
        kernel = np.zeros(4 * 2**level + 1)
        kernel[::2**level] = np.array([1, 4, 6, 4, 1]) / 16
        low_x = ndimage.convolve1d(smooth, kernel, axis=1, mode="reflect")
        low_y = ndimage.convolve1d(smooth, kernel, axis=0, mode="reflect")
        low_xy = ndimage.convolve1d(low_x, kernel, axis=0, mode="reflect")
        yield 0.0, low_x - low_xy
        yield pi / 2, low_y - low_xy
        yield pi / 4, smooth - low_x - low_y + low_xy
        smooth = low_xy
    yield None, smooth


def _frequency_grid(shape):
    return fft.fftfreq(shape[1])[None, :], fft.fftfreq(shape[0])[:, None]


def log_gabor_components(image, p, cancel=None):
    """Zero-DC log-Gaussian radial filters with axial angular weighting."""
    angles = _angles(p)
    if not 0.03 <= p.first_frequency <= 0.45 or not 0.1 <= p.bandwidth <= 1.5:
        raise ValueError("Invalid Log-Gabor central frequency or bandwidth")
    spectrum = fft.fft2(image)
    x, y = _frequency_grid(image.shape)
    radius = np.hypot(x, y)
    phase = np.arctan2(y, x)
    for level in range(p.levels):
        center = p.first_frequency / 2**level
        radial = np.exp(-0.5 * (np.log(np.maximum(radius, 1e-30) / center) / p.bandwidth)**2)
        radial[0, 0] = 0
        for angle in angles:
            check_cancel(cancel)
            difference = (phase - angle - pi / 2 + pi / 2) % pi - pi / 2
            angular = np.exp(-0.5 * (difference / np.deg2rad(p.angle_step_deg))**2)
            # Signed response keeps crest/trough locations visible in the maps.
            yield angle, fft.ifft2(spectrum * radial * angular).real


def _shear_windows(shape, levels):
    x, y = _frequency_grid(shape)
    radius = np.maximum(abs(x), abs(y))
    yield None, np.exp(-0.5 * (radius / (0.32 / 2**levels))**4)
    for level in range(levels):
        center = 0.32 / 2**(levels - 1 - level)
        radial = np.exp(-0.5 * (np.log2(np.maximum(radius, 1e-30) / center) / 0.72)**2)
        count = ceil(2**(level / 2))
        for horizontal in (True, False):
            major, minor = (x, y) if horizontal else (y, x)
            slope = np.divide(minor, major, out=np.zeros(shape), where=major != 0)
            cone = np.clip((1.15 - abs(slope)) / 0.3, 0, 1) * (major != 0)
            for shear in range(-count, count + 1):
                parameter = shear / count
                axis = np.arctan2(parameter, 1) if horizontal else np.arctan2(1, parameter)
                weight = cone * np.maximum(1 - abs(count * slope - shear), 0)
                yield (axis + pi / 2) % pi, radial * weight


def shearlet_components(image, levels, cancel=None):
    """Experimental sampled cone frame with normalized Fourier reconstruction.

    Yields (tangent, coefficient, synthesis window). Windows use dyadic radial
    scales, parabolic angular subdivision and discrete shears on both cones.
    """
    if not 1 <= levels <= 5:
        raise ValueError("Shearlet levels must be in [1, 5]")
    spectrum = fft.fft2(image)
    energy = np.zeros(image.shape)
    for _, window in _shear_windows(image.shape, levels):
        check_cancel(cancel)
        energy += window**2
    if (energy <= 0).any():
        raise ValueError("Shearlet windows do not cover the Fourier grid")
    denominator = np.sqrt(energy)
    for angle, window in _shear_windows(image.shape, levels):
        check_cancel(cancel)
        normalized = window / denominator
        yield angle, fft.ifft2(spectrum * normalized), normalized


def reconstruct_shearlets(components):
    spectrum = None
    for _, coefficients, window in components:
        term = fft.fft2(coefficients) * window
        spectrum = term if spectrum is None else spectrum + term
    if spectrum is None:
        raise ValueError("At least one shearlet component is required")
    return fft.ifft2(spectrum).real


def response_map(dem, p, algorithm, *, progress=None, cancel=None):
    """Return maximum band strength and its axial tangent; mask NoData margins."""
    if not 1 <= p.levels <= 5 or not 0 < p.threshold <= 1:
        raise ValueError("Invalid response levels or threshold")
    check_cancel(cancel)
    if algorithm == "log-gabor" and not 0.03 <= p.first_frequency <= 0.45:
        raise ValueError("Invalid Log-Gabor central frequency")
    image, valid = _filled(dem)
    image = image - float(image[valid].mean()) if valid.any() else image
    if algorithm == "directional":
        components = directional_components(image, p, cancel)
        radius = ceil(4 * p.first_scale_px * 2**(p.levels - 1))
    elif algorithm == "wavelet":
        components = wavelet_components(image, p.levels, cancel)
        radius = 2 * (2**p.levels - 1)
    elif algorithm == "log-gabor":
        components = log_gabor_components(image, p, cancel)
        radius = ceil(2**(p.levels - 1) / p.first_frequency)
    elif algorithm == "shearlet":
        components = ((a, c) for a, c, _ in shearlet_components(image, p.levels, cancel))
        radius = 2**p.levels
    else:
        raise ValueError(f"Unknown filter bank: {algorithm}")
    if algorithm in ("log-gabor", "shearlet"):
        # Reflect margins to soften FFT wrap-around, retaining original raster coordinates.
        padding = min(radius, max(image.shape))
        image = np.pad(image, padding, mode="reflect")
        if algorithm == "log-gabor":
            components = log_gabor_components(image, p, cancel)
        else:
            components = ((a, c) for a, c, _ in shearlet_components(image, p.levels, cancel))
        crop = np.s_[padding:-padding, padding:-padding]
    else:
        crop = np.s_[:, :]
    strength = np.zeros(valid.shape)
    tangent = np.zeros_like(strength)
    for index, (angle, band) in enumerate(components):
        check_cancel(cancel)
        if angle is None:
            continue
        amplitude = abs(band[crop])
        stronger = amplitude > strength
        strength[stronger] = amplitude[stronger]
        if algorithm == "wavelet" and angle == pi / 4:
            # Diagonal detail alone has ambiguous sign/direction; estimate a
            # tangent from the local derivative of this coefficient.
            if min(amplitude.shape) > 1:
                gy, gx = np.gradient(band[crop])
                tangent[stronger] = np.arctan2(gx[stronger], -gy[stronger]) % pi
        else:
            tangent[stronger] = angle
        if progress:
            progress(min(70, 5 + index))
    safe = detector_support_mask(valid, radius=radius)
    strength[~safe] = 0
    floor = np.max(abs(image), initial=0) * 1e-10
    positive = strength[strength > floor]
    if positive.size:
        strength = np.minimum(strength / np.percentile(positive, 97), 1)
    else:
        strength.fill(0)
    return Response(strength, tangent, safe)


class FilterDetector:
    """Shared nonmaxima-suppressed ridge linking for four distinct transforms."""

    def __init__(self, algorithm_id):
        self.algorithm_id = algorithm_id

    def run(self, dem, parameters, *, progress=None, cancel=None):
        p = parameters
        if not 2 <= p.minimum_length_px <= 512 or not 1 <= p.max_lines <= 512:
            raise ValueError("Invalid filter extraction limits")
        if progress:
            progress(0)
        response = response_map(dem, p, self.algorithm_id, progress=progress, cancel=cancel)
        score, tangent, valid = response.strength, response.tangent_rad, response.valid
        y, x = np.nonzero(score >= p.threshold)
        angles = tangent[y, x]
        normal_y, normal_x = np.cos(angles), -np.sin(angles)
        ridge = np.zeros_like(score)
        peak = score[y, x]
        for sign in (-1, 1):
            side = ndimage.map_coordinates(
                score, (y + sign * normal_y, x + sign * normal_x), order=1, mode="nearest"
            )
            peak = np.where(peak >= side - 1e-9, peak, 0)
        ridge[y, x] = peak
        y, x = np.nonzero(ridge)
        order = np.argsort(-ridge[y, x], kind="stable")
        occupied = np.zeros_like(valid)
        trace = TraceParameters(min_length_px=p.minimum_length_px,
                                max_steps=max(192, p.minimum_length_px * 2),
                                support_threshold=p.threshold, max_gap_px=2)
        paths, confidences = [], []
        attempts = 0
        for i in order:
            if occupied[y[i], x[i]]:
                continue
            check_cancel(cancel)
            if attempts >= 20 * p.max_lines or len(paths) >= p.max_lines:
                break
            attempts += 1
            path = _trace((y[i], x[i]), (ridge, tangent, valid), trace, occupied, cancel)
            if len(path) < p.minimum_length_px:
                continue
            px, py = np.asarray(path).T
            support = float(np.mean(ridge[py, px] >= p.threshold))
            if support < 0.65:
                continue
            paths.append(path)
            confidences.append(support)
            occupied[py, px] = True
        result = build_path_result(
            dem, paths, algorithm_id=self.algorithm_id, parameters=p, confidences=confidences,
        )
        if progress:
            progress(100)
        return result

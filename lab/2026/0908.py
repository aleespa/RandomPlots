"""
Vorticity Tide

Every swirl here is the 2D Navier-Stokes equations being solved, frame by
frame:

    ∂ω/∂t + u·∇ω = ν∇²ω + f

A fluid stirred by a few slow travelling waves, shown through its vorticity ω:
clockwise eddies in blue, counter-clockwise in amber, calm water black. The
filaments and mergers are the non-linear term u·∇ω at work, not a texture.

The loop is a second genuine solution that starts where the first ended and is
gently nudged back onto it, so the last frame flows into the first.

Posted on the day an AI-generated proof of the Navier-Stokes Millennium Prize
problem was announced.

Made with Python, torch and matplotlib. Pseudo-spectral solver, 60 fps.

#generativeart #creativecoding #navierstokes #fluiddynamics #turbulence
#mathart #python #vorticity #simulation
"""

import gc
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap

from common.image_processing import ImageProcessingSettings

# RP_FPS=5 in the environment gives a quick 40-frame test loop with the same timing.
FPS = int(os.environ.get("RP_FPS", "60"))
LOOP_SECONDS = 8
LOOP_FRAMES = FPS * LOOP_SECONDS  # exact loop period, in frames
REPEATS = 3  # play the rendered loop this many times back-to-back in the mp4
WIDTH, HEIGHT = 1080, 1920  # 9:16 for Reels/Stories
SIM_WIDTH, SIM_HEIGHT = 540, 960  # the solver's grid; frames are spectrally upsampled
LX, LY = 2 * np.pi * SIM_WIDTH / SIM_HEIGHT, 2 * np.pi  # periodic box, square cells

# Diverging vorticity ramp: ice white -> sky -> steel blue -> navy -> black at
# zero -> oxblood -> rust -> amber -> cream. Negative (clockwise) vortices read
# cold, positive ones warm, and the still fluid between them stays black.
VORTICITY_TIDE = [
    "#f4fbff",
    "#bfe4f7",
    "#3f8fc4",
    "#123a5c",
    "#05101c",
    "#000000",
    "#160804",
    "#3d1206",
    "#b03a08",
    "#ffb347",
    "#fff4e2",
]

PNG_WRITERS = 4  # threads encoding PNGs while the GPU renders the next frame


def _device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.warning("CUDA not available -- solving on the CPU with torch")
    return torch.device("cpu")


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = int(np.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def _blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) tensor with reflected borders."""
    radius = (kernel.numel() - 1) // 2
    channels = image.shape[0]
    x = image[None]
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(
        x, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels
    )
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(
        x, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels
    )
    return x[0]


class _Solver:
    """
    Pseudo-spectral 2D Navier-Stokes in vorticity-streamfunction form on a
    periodic box, integrated with RK4 and an exact integrating factor for the
    linear (viscous + drag) terms. Everything lives on one torch device.
    """

    def __init__(self, knobs: dict, rng: np.random.Generator, device: torch.device):
        self.k = knobs
        self.device = device
        nx, ny = SIM_WIDTH, SIM_HEIGHT
        self.dx = LX / nx

        kx = 2 * np.pi * np.fft.rfftfreq(nx, LX / nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, LY / ny)
        self.kx = torch.tensor(kx, dtype=torch.float32, device=device)[None, :]
        self.ky = torch.tensor(ky, dtype=torch.float32, device=device)[:, None]
        k2 = self.kx**2 + self.ky**2
        self.inv_k2 = torch.where(
            k2 > 0, 1.0 / k2.clamp(min=1e-12), torch.zeros_like(k2)
        )
        # 2/3-rule dealiasing of the quadratic term.
        self.mask = (
            (self.kx.abs() < (2.0 / 3.0) * kx.max())
            & (self.ky.abs() < (2.0 / 3.0) * abs(ky).max())
        ).to(torch.float32)
        # Linear decay rate: Newtonian viscosity, a touch of hyperviscosity to
        # keep the smallest filaments crisp, and a large-scale drag so the
        # forced flow reaches a statistically steady state.
        self.decay = knobs["nu"] * k2 + knobs["nu4"] * k2**2 + knobs["drag"]
        self.decay[0, 0] = 0.0

        x = torch.arange(nx, device=device, dtype=torch.float32) * self.dx
        y = torch.arange(ny, device=device, dtype=torch.float32) * self.dx
        self.X = x[None, :]
        self.Y = y[:, None]

        # Forcing: a handful of travelling vorticity waves at low wavenumber.
        # Each completes an integer number of cycles per loop period, which is
        # what makes the driven flow (and thus the clip) periodic in time.
        candidates = [
            (i, j)
            for i in range(0, 8)
            for j in range(-8, 9)
            if (i > 0 or j > 0)
            and knobs["k_min"]
            <= np.hypot(i * 2 * np.pi / LX, j * 2 * np.pi / LY)
            <= knobs["k_max"]
        ]
        picks = rng.choice(len(candidates), size=knobs["n_modes"], replace=False)
        modes = []
        for p in picks:
            i, j = candidates[p]
            modes.append(
                (
                    i * 2 * np.pi / LX,
                    j * 2 * np.pi / LY,
                    float(rng.uniform(0, 2 * np.pi)),
                    int(rng.choice([-2, -1, 1, 2])),
                    knobs["force"] * float(rng.uniform(0.6, 1.0)),
                )
            )
        self.modes = modes
        self._prepare_forcing()

        # Nudging acts on |kx|, |ky| <= k_nudge; that block of the rfft grid is
        # rows [-kc, kc] (wrapped) and the first few columns.
        kc = int(knobs["k_nudge"])
        ncol = int(np.floor(kc / (2 * np.pi / LX))) + 1
        rows = torch.cat([torch.arange(0, kc + 1), torch.arange(ny - kc, ny)]).to(
            device
        )
        self.nudge_slice = (rows[:, None], torch.arange(ncol, device=device)[None, :])
        self.nudge_target = None
        self.nudge_lambda = 0.0
        logger.info(
            "forcing modes (kx, ky, cycles): "
            + ", ".join(f"({m[0]:.2f}, {m[1]:.2f}, {m[3]:+d})" for m in modes)
        )

    # --- spectral helpers ---------------------------------------------------
    def to_hat(self, w):
        return torch.fft.rfft2(w)

    def to_phys(self, w_hat):
        return torch.fft.irfft2(w_hat, s=(SIM_HEIGHT, SIM_WIDTH))

    def velocity(self, w_hat):
        psi_hat = w_hat * self.inv_k2  # omega = -laplacian(psi)
        u = self.to_phys(1j * self.ky * psi_hat)
        v = self.to_phys(-1j * self.kx * psi_hat)
        return u, v

    def _prepare_forcing(self):
        """
        Spectral templates so the forcing costs a few axpys per stage, not FFTs:
        cos(theta + d) = cos(theta) cos(d) - sin(theta) sin(d) with theta the
        static wave and d the travelling phase.
        """
        self.force_cos, self.force_sin = [], []
        for kx, ky, phase, _, amp in self.modes:
            theta = kx * self.X + ky * self.Y + phase
            self.force_cos.append(amp * self.to_hat(torch.cos(theta)))
            self.force_sin.append(amp * self.to_hat(torch.sin(theta)))

    def forcing_hat(self, t: float):
        omega = 2 * np.pi / self.k["loop_time"]
        f = torch.zeros_like(self.force_cos[0])
        for (_, _, _, cycles, _), c, s in zip(
            self.modes, self.force_cos, self.force_sin
        ):
            d = cycles * omega * t
            f += np.cos(d) * c - np.sin(d) * s
        return f

    def rhs(self, w_hat, t: float):
        """Non-linear advection (dealiased) plus forcing; the linear part is exact."""
        u, v = self.velocity(w_hat)
        wx = self.to_phys(1j * self.kx * w_hat)
        wy = self.to_phys(1j * self.ky * w_hat)
        rhs = -self.to_hat(u * wx + v * wy) * self.mask + self.forcing_hat(t)
        if self.nudge_target is not None:
            # Relax the large scales toward the reference trajectory.
            rhs[self.nudge_slice] += self.nudge_lambda * (
                self.nudge_target - w_hat[self.nudge_slice]
            )
        return rhs

    def step(self, w_hat, t: float, dt: float):
        """One RK4 step with integrating factor exp(-decay t) for the linear terms."""
        e1 = torch.exp(-self.decay * dt / 2)
        e2 = e1 * e1
        k1 = self.rhs(w_hat, t)
        k2 = self.rhs(e1 * (w_hat + 0.5 * dt * k1), t + 0.5 * dt)
        k3 = self.rhs(e1 * w_hat + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.rhs(e2 * w_hat + dt * e1 * k3, t + dt)
        return e2 * w_hat + dt / 6 * (e2 * k1 + 2 * e1 * (k2 + k3) + k4)

    def advance(self, w_hat, t: float, duration: float, nudge=None):
        """
        Advance by `duration`, picking substeps from the CFL condition.

        `nudge`, if given, is (lambda_fn, target_fn): the relaxation rate and the
        low-wavenumber reference spectrum as functions of the local phase in
        [0, 1] across this call.
        """
        u, v = self.velocity(w_hat)
        umax = float(torch.maximum(u.abs().max(), v.abs().max()).item()) + 1e-6
        substeps = max(1, int(np.ceil(duration * umax / (self.k["cfl"] * self.dx))))
        dt = duration / substeps
        for i in range(substeps):
            if nudge is not None:
                phase = (i + 0.5) / substeps
                self.nudge_lambda = nudge[0](phase)
                self.nudge_target = nudge[1](phase)
            w_hat = self.step(w_hat, t + i * dt, dt)
        self.nudge_target = None
        return w_hat, substeps, umax

    def low_modes(self, w_hat):
        """The cropped block of spectral coefficients that nudging acts on."""
        return w_hat[self.nudge_slice].clone()

    def initial(self, gen: torch.Generator):
        """Weak random vorticity at the forcing scales to seed the instabilities."""
        w = torch.randn((SIM_HEIGHT, SIM_WIDTH), generator=gen, device=self.device)
        w_hat = self.to_hat(w)
        k = torch.sqrt(self.kx**2 + self.ky**2)
        w_hat = w_hat * torch.exp(-((k / 6.0) ** 2))
        return w_hat * (0.05 / self.to_phys(w_hat).abs().max())

    def upsample(self, w_hat):
        """Spectral zero-padding from the solver grid to the output frame -- exact."""
        big = torch.zeros(
            (HEIGHT, WIDTH // 2 + 1), dtype=w_hat.dtype, device=self.device
        )
        half = SIM_HEIGHT // 2
        cols = SIM_WIDTH // 2 + 1
        big[:half, :cols] = w_hat[:half]
        big[-half:, :cols] = w_hat[-half:]
        big *= (HEIGHT * WIDTH) / (SIM_HEIGHT * SIM_WIDTH)
        return torch.fft.irfft2(big, s=(HEIGHT, WIDTH))


class _Painter:
    """Signed vorticity -> diverging ramp with a soft glow, as a BGR uint8 frame."""

    def __init__(self, knobs: dict, device: torch.device):
        self.k = knobs
        lut = LinearSegmentedColormap.from_list("tide", VORTICITY_TIDE, N=2048)(
            np.linspace(0.0, 1.0, 2048)
        )[:, :3]
        self.lut = torch.tensor(lut, dtype=torch.float32, device=device)
        self.glow_kernel = _gaussian_kernel(knobs["glow_sigma"], device)
        self.scale = 1.0

    def compose(self, w: torch.Tensor) -> np.ndarray:
        level = torch.tanh(self.k["gain"] * w / self.scale)  # (-1, 1)
        index = ((level + 1.0) * 0.5 * (self.lut.shape[0] - 1)).long()
        rgb = self.lut[index].permute(2, 0, 1)  # (3, H, W)
        if self.k["glow_weight"] > 0:
            lift = (level.abs() - self.k["glow_floor"]).clamp(0.0, 1.0)
            rgb = rgb + self.k["glow_weight"] * _blur(rgb * lift, self.glow_kernel)
        frame = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return frame.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _write_png(path, frame: np.ndarray):
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write {path}")


def generate(settings: ImageProcessingSettings = None):
    """
    Vorticity Tide -- the two-dimensional incompressible Navier-Stokes equations

        d(omega)/dt + u . grad(omega) = nu laplacian(omega) - alpha omega + f,
        u = (d psi/dy, -d psi/dx),   omega = -laplacian(psi),

    solved directly and rendered as an exactly looping 9:16 clip, on the day an
    AI-generated proof of the Navier-Stokes existence-and-smoothness Millennium
    problem was announced. The picture is the vorticity field: clockwise
    vortices in cold blues, counter-clockwise ones in warm ambers, and the
    quiet fluid between them black. Every filament, roll-up and merger on
    screen is the non-linear term u . grad(omega) doing its work.

    Numerics: pseudo-spectral on a periodic 540 x 960 box with 2/3-rule
    dealiasing, RK4 in time with an exact integrating factor for viscosity,
    hyperviscosity and large-scale drag, and adaptive substeps from the CFL
    condition. Frames are upsampled to 1080 x 1920 by spectral zero-padding,
    which is exact for the band-limited field.

    The loop: the flow is driven by a few travelling vorticity waves at low
    wavenumber, each completing an integer number of cycles per loop period,
    so the forcing is exactly periodic. After a warm-up, one reference period
    A(t) is recorded. The clip is then a second, genuine Navier-Stokes
    trajectory that starts where A ended and is *nudged* back onto A: a
    relaxation term lambda(t) (A_k - omega_k) is added to the large-scale
    modes only (|k| <= k_nudge), with lambda ramping up from zero. Nudging the
    large scales of 2D Navier-Stokes is enough to synchronise the whole field
    (the Olson-Titi data-assimilation result), so the small scales fall into
    step through the dynamics themselves -- there is only ever one velocity
    field advecting one vorticity field, and no second fluid ghosts through.
    Whatever mismatch survives is removed by a blend over the last few
    percent of the loop, so frame N joins frame 0 exactly; the residual is
    logged so it can be judged.

    Everything numerical runs on the GPU with torch: FFTs, the RK4 stages, the
    palette lookup and the glow. Only the finished 8-bit frame crosses back to
    the host, where a small thread pool encodes the PNGs.
    """
    settings = settings or ImageProcessingSettings(1)
    rng = settings.rng

    # --- knobs ---
    knobs = dict(
        loop_time=8.0,  # simulated time units per loop; more = faster swirl
        warmup_periods=3,  # loop periods run before recording
        nu=1.5e-5,  # Newtonian viscosity; lower = finer filaments
        nu4=4e-10,  # hyperviscosity, cleans the grid scale
        drag=0.05,  # large-scale linear drag; sets the saturated amplitude
        force=0.8,  # forcing amplitude per wave, in vorticity units
        n_modes=5,  # number of travelling forcing waves
        k_min=1.5,  # forcing wavenumber band (physical |k|; box height is 2 pi)
        k_max=3.6,
        cfl=0.45,
        k_nudge=48,  # nudging acts on |k| <= this; the rest synchronises on its own
        nudge_lambda=4.0,  # peak relaxation rate toward the reference, per time unit
        nudge_ramp=0.4,  # fraction of the loop over which the rate ramps up
        join_fraction=0.08,  # tail of the loop blended onto the reference exactly  # RK4 on spectral advection needs this well under 1
        gain=1.2,  # steepness of the tanh tone curve
        glow_weight=0.35,
        glow_sigma=7.0,
        glow_floor=0.55,  # |level| above which a pixel starts to glow
    )
    exposure_pct = 99.0  # |vorticity| percentile that reaches the ramp's ends

    device = _device()
    solver = _Solver(knobs, rng, device)
    painter = _Painter(knobs, device)
    gen = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**31)))

    dt_frame = knobs["loop_time"] / LOOP_FRAMES
    w_hat = solver.initial(gen)
    t = 0.0

    # --- warm-up onto the driven flow's attractor ---
    warm_frames = knobs["warmup_periods"] * LOOP_FRAMES
    logger.info(
        f"warming up for {knobs['warmup_periods']} periods ({warm_frames} frame-steps)"
    )
    for i in range(warm_frames):
        w_hat, substeps, umax = solver.advance(w_hat, t, dt_frame)
        t += dt_frame
        if i % LOOP_FRAMES == 0:
            w = solver.to_phys(w_hat)
            logger.info(
                f"  t={t:7.2f}  |omega|max={w.abs().max().item():.2f}  umax={umax:.2f}  substeps={substeps}"
            )

    # Fixed exposure for the whole clip, measured at the end of the warm-up.
    w = solver.to_phys(w_hat)
    painter.scale = float(
        torch.quantile(w.abs().flatten()[::5], exposure_pct / 100).item()
    )
    logger.info(f"vorticity scale {painter.scale:.3f}")

    # --- reference period A: the trajectory the loop will return to ---
    logger.info(f"recording reference period ({LOOP_FRAMES} frames)")
    ref_low = []  # low-mode spectra at loop phase i / N, i = 0..N
    ref_full = []  # full fields at the same phases, for the final join
    for i in range(LOOP_FRAMES + 1):
        ref_low.append(solver.low_modes(w_hat))
        ref_full.append(solver.upsample(w_hat).to(torch.float16).cpu())
        if i < LOOP_FRAMES:
            w_hat, _, _ = solver.advance(w_hat, t, dt_frame)
            t += dt_frame
    # w_hat is now A(T) = the loop's first frame, and ref_*[N] is that same state.

    # --- the loop: NS driven back onto A by nudging its large scales ---
    # Frame i of the loop is the nudged trajectory at phase i / N, starting from
    # A(T). The relaxation rate ramps up over the first `nudge_ramp` of the
    # loop and then holds, so the flow is untouched near the join and
    # synchronised with A(t) well before the end. What little mismatch remains
    # is removed by a short full-field blend over the last `join_frames`.
    lam_max = knobs["nudge_lambda"]
    ramp = knobs["nudge_ramp"]

    def lam(phase: float) -> float:
        x = min(phase / ramp, 1.0)
        return lam_max * x * x * (3.0 - 2.0 * x)

    join_frames = max(2, int(round(knobs["join_fraction"] * LOOP_FRAMES)))
    frames_path = settings.frames_path
    logger.info(f"rendering the nudged loop ({LOOP_FRAMES} frames)")
    residuals = []
    with ThreadPoolExecutor(PNG_WRITERS) as pool:
        pending = []
        for i in range(LOOP_FRAMES):
            w_loop = solver.upsample(w_hat)
            w_ref = ref_full[i].to(device, torch.float32)
            residual = ((w_loop - w_ref).norm() / w_ref.norm()).item()
            if i % (LOOP_FRAMES // 8) == 0 or i == LOOP_FRAMES - 1:
                residuals.append(residual)
            k_join = i - (LOOP_FRAMES - join_frames)
            if k_join >= 0:
                sj = (k_join + 1) / (join_frames + 1)
                sj = sj * sj * (3.0 - 2.0 * sj)
                w_loop = (1.0 - sj) * w_loop + sj * w_ref
            frame = painter.compose(w_loop)
            pending.append(
                pool.submit(_write_png, frames_path / f"frame{i:04d}.png", frame)
            )
            if len(pending) > 2 * PNG_WRITERS:
                pending.pop(0).result()

            lo, hi = ref_low[i], ref_low[i + 1]
            frame_phase = i / LOOP_FRAMES
            nudge = (
                lambda ph, f=frame_phase: lam(f + ph / LOOP_FRAMES),
                lambda ph, lo=lo, hi=hi: lo + ph * (hi - lo),
            )
            w_hat, _, _ = solver.advance(w_hat, t, dt_frame, nudge)
            t += dt_frame
        for job in pending:
            job.result()
    logger.info(
        "loop-to-reference residual |L - A| / |A| at 8 phases + last frame: "
        + ", ".join(f"{r:.3f}" for r in residuals)
    )

    for repeat in range(1, REPEATS):
        for i in range(LOOP_FRAMES):
            shutil.copyfile(
                frames_path / f"frame{i:04d}.png",
                frames_path / f"frame{repeat * LOOP_FRAMES + i:04d}.png",
            )

    del solver, painter, ref_low, ref_full, w_hat
    gc.collect()
    settings.save_video(FPS, crf=18)


if __name__ == "__main__":
    generate()

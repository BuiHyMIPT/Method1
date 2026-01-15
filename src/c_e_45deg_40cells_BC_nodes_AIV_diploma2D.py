
import numpy as np
import matplotlib.pyplot as plt


nx_cells, ny_cells = 40, 40
nx, ny = nx_cells + 1, ny_cells + 1

x_min, x_max = -1.0, 1.0   # x=0 is inside
y_min, y_max = 0.0, 1.0

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
dx = (x_max - x_min) / nx_cells
dy = (y_max - y_min) / ny_cells
X, Y = np.meshgrid(x, y, indexing="ij")  # (nx, ny)

# -----------------------------
# Velocity: 45 degrees, constant magnitude
# -----------------------------
speed = 0.6523404
u0 = speed / np.sqrt(2.0)
v0 = speed / np.sqrt(2.0)

# -----------------------------
# Initial / boundary data
# -----------------------------
rho_L = 1.740648
rho_R = 1.0

def apply_bc(rho: np.ndarray) -> None:
    rho[:, 0] = rho[:, 1]
    rho[:, -1] = rho[:, -2]

    rho[0, :] = rho_L
    rho[-1, :] = rho_R


Lx = np.full((nx, ny), dx)  # horizontal face length (in x-direction)
Ly = np.full((nx, ny), dy)  # vertical face length (in y-direction)

# Half control volumes on domain boundaries
Lx[0, :] *= 0.5
Lx[-1, :] *= 0.5
Ly[:, 0] *= 0.5
Ly[:, -1] *= 0.5

V = Lx * Ly  # area of nodal dual CV


def compute_fluxes_centered(rho: np.ndarray, nu: np.ndarray):
    F = np.zeros((nx + 1, ny))
    G = np.zeros((nx, ny + 1))

    for i in range(nx - 1):
        rho_face = 0.5 * (rho[i, :] + rho[i + 1, :])
        nu_face = 0.5 * (nu[i, :] + nu[i + 1, :])
        grad = (rho[i + 1, :] - rho[i, :]) / dx
        F[i + 1, :] = u0 * rho_face - nu_face * grad

    rho_face = 0.5 * (rho_L + rho[0, :])
    nu_face = nu[0, :]
    grad = (rho[0, :] - rho_L) / (0.5 * dx)
    F[0, :] = u0 * rho_face - nu_face * grad

    rho_face = 0.5 * (rho[-1, :] + rho_R)
    nu_face = nu[-1, :]
    grad = (rho_R - rho[-1, :]) / (0.5 * dx)
    F[-1, :] = u0 * rho_face - nu_face * grad

    for j in range(ny - 1):
        rho_face = 0.5 * (rho[:, j] + rho[:, j + 1])
        nu_face = 0.5 * (nu[:, j] + nu[:, j + 1])
        grad = (rho[:, j + 1] - rho[:, j]) / dy
        G[:, j + 1] = v0 * rho_face - nu_face * grad

    rho_face = rho[:, 0]  # equal to rho[:,1] after BC
    G[:, 0] = v0 * rho_face
    # North face index ny
    rho_face = rho[:, -1]
    G[:, -1] = v0 * rho_face

    return F, G


def step_once(rho: np.ndarray, dt: float, nu: np.ndarray) -> np.ndarray:
    m = rho * V
    F, G = compute_fluxes_centered(rho, nu)

    m_new = m.copy()
    for i in range(nx):
        for j in range(ny):
            flux_x = (F[i + 1, j] - F[i, j]) * Ly[i, j]
            flux_y = (G[i, j + 1] - G[i, j]) * Lx[i, j]
            m_new[i, j] = m[i, j] - dt * (flux_x + flux_y)

    rho_new = m_new / V
    apply_bc(rho_new)
    return rho_new


def monotonicity_violations(rho_old: np.ndarray, rho_new: np.ndarray, tol: float = 1e-12):
    viol = np.zeros_like(rho_new, dtype=bool)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            local = [
                rho_old[i, j],
                rho_old[i - 1, j], rho_old[i + 1, j],
                rho_old[i, j - 1], rho_old[i, j + 1],
            ]
            mn, mx = min(local), max(local)
            if (rho_new[i, j] < mn - tol) or (rho_new[i, j] > mx + tol):
                viol[i, j] = True
    return viol


def step_with_aiv(rho: np.ndarray, dt: float,
                  beta_limiter: float = 0.2,
                  const_rho: float = 0.05,
                  max_inner_iters: int = 30):
    h = min(dx, dy)
    beta = np.zeros_like(rho)

    rho_old = rho.copy()
    apply_bc(rho_old)

    rho_new = None

    for _ in range(max_inner_iters):
        nu = beta * (h * h) / dt
        rho_new = step_once(rho_old, dt, nu)

        viol = monotonicity_violations(rho_old, rho_new)

        if not np.any(viol):
            break

        # Increase beta locally (and a bit around) where monotonicity is broken
        idx = np.argwhere(viol)
        for i, j in idx:
            if beta[i, j] < beta_limiter:
                beta[i, j] = min(beta_limiter, beta[i, j] + const_rho)
            # also touch immediate neighbors (helps in 2D)
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ii, jj = i+di, j+dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    if beta[ii, jj] < beta_limiter:
                        beta[ii, jj] = min(beta_limiter, beta[ii, jj] + 0.5*const_rho)

        # keep boundaries beta=0 (Dirichlet/Neumann dominate there)
        beta[0, :] = 0.0
        beta[-1, :] = 0.0
        beta[:, 0] = 0.0
        beta[:, -1] = 0.0

    # Final rho_new from last iteration
    return rho_new, beta


def run_sim(use_aiv: bool, T_end: float = 0.8, CFL: float = 0.35):
    # Initial condition: front at x=0
    rho0 = np.where(X < 0.0, rho_L, rho_R).astype(float)
    apply_bc(rho0)

    # dt based on CFL for advection
    dt = CFL * min(dx / abs(u0), dy / abs(v0))
    nsteps = int(np.ceil(T_end / dt))
    dt = T_end / nsteps

    rho = rho0.copy()
    beta_last = None

    for _ in range(nsteps):
        if use_aiv:
            rho, beta_last = step_with_aiv(rho, dt,
                                           beta_limiter=0.2,
                                           const_rho=0.05,
                                           max_inner_iters=30)
        else:
            nu0 = np.zeros_like(rho)
            rho = step_once(rho, dt, nu0)

    return rho, beta_last, dt, nsteps



rho_no, _, dt, nsteps = run_sim(use_aiv=False)
rho_aiv, beta_last, _, _ = run_sim(use_aiv=True)

print(f"dt={dt:.6g}, nsteps={nsteps}")
if beta_last is not None:
    print(f"beta_last: min={beta_last.min():.3g}, max={beta_last.max():.3g}, mean={beta_last.mean():.3g}")

# 2D comparison heatmaps (same color scale)
vmin = float(min(rho_no.min(), rho_aiv.min()))
vmax = float(max(rho_no.max(), rho_aiv.max()))

fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

im0 = ax[0].imshow(
    rho_no.T, origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    aspect="auto", vmin=vmin, vmax=vmax
)
ax[0].set_title("No AIV (centered flux)")
ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
cbar0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
cbar0.set_label("Плотность ρ")

im1 = ax[1].imshow(
    rho_aiv.T, origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    aspect="auto", vmin=vmin, vmax=vmax
)
ax[1].set_title("With AIV (adaptive β)")
ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
cbar1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
cbar1.set_label("Плотность ρ")

plt.suptitle("Density field ρ(x,y) — grid 40×40 cells, angle 45°")
plt.show()

# 3D surfaces + colorbar label
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 3D: No AIV
fig = plt.figure(figsize=(10, 6))
ax3 = fig.add_subplot(111, projection="3d")
surf0 = ax3.plot_surface(
    X, Y, rho_no,
    cmap="viridis", linewidth=0, antialiased=True,
    vmin=vmin, vmax=vmax
)
ax3.set_title("3D surface: ρ(x,y) — No AIV")
ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("ρ")
cbar3d0 = fig.colorbar(surf0, ax=ax3, shrink=0.6, pad=0.1)
cbar3d0.set_label("Плотность ρ")
plt.show()

# 3D: With AIV
fig = plt.figure(figsize=(10, 6))
ax3 = fig.add_subplot(111, projection="3d")
surf1 = ax3.plot_surface(
    X, Y, rho_aiv,
    cmap="viridis", linewidth=0, antialiased=True,
    vmin=vmin, vmax=vmax
)
ax3.set_title("3D surface: ρ(x,y) — With AIV")
ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("ρ")
cbar3d1 = fig.colorbar(surf1, ax=ax3, shrink=0.6, pad=0.1)
cbar3d1.set_label("Плотность ρ")
plt.show()

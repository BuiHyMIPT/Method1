import numpy as np
import matplotlib.pyplot as plt

nx_cells, ny_cells = 40, 40
nx, ny = nx_cells + 1, ny_cells + 1

x_min, x_max = -1.0, 1.0
y_min, y_max = 0.0, 1.0

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
dx = (x_max - x_min) / nx_cells
dy = (y_max - y_min) / ny_cells

X, Y = np.meshgrid(x, y, indexing="ij")

# -----------------------------
# Velocity: 45 degrees, constant magnitude
# -----------------------------
speed = 0.6523404
u0 = speed / np.sqrt(2.0)
v0 = speed / np.sqrt(2.0)

# -----------------------------
# Initial / boundary conditions
# -----------------------------
rho_L = 1.740648
rho_R = 1.0

def apply_bc(rho: np.ndarray) -> None:
    rho[:, 0]  = rho[:, 1]
    rho[:, -1] = rho[:, -2]
    rho[0, :]  = rho_L
    rho[-1, :] = rho_R

# Initial front at x=0
rho0 = np.where(X < 0.0, rho_L, rho_R).astype(float)
apply_bc(rho0)

# -----------------------------
# Dual control volume geometry (node-based)
# -----------------------------
Lx = np.full((nx, ny), dx)
Ly = np.full((nx, ny), dy)

# half lengths on boundaries
Lx[0, :]  *= 0.5
Lx[-1, :] *= 0.5
Ly[:, 0]  *= 0.5
Ly[:, -1] *= 0.5

V = Lx * Ly  # area per node dual control volume


def compute_cfl(dt: float) -> float:
    return dt * (abs(u0) / dx + abs(v0) / dy)

dt = 0.004
CFL = compute_cfl(dt)
print(f"dt={dt:.6e}, CFL={CFL:.6f}")

def compute_fluxes_centered(rho_for_flux: np.ndarray, nu: np.ndarray):
    """
    F = rho*u - nu * d(rho)/dx
    G = rho*v - nu * d(rho)/dy
    """
    F = np.zeros((nx + 1, ny))
    G = np.zeros((nx, ny + 1))

    # x-faces i+1/2
    rho_face = 0.5 * (rho_for_flux[1:, :] + rho_for_flux[:-1, :])
    nu_face  = 0.5 * (nu[1:, :] + nu[:-1, :])
    drho_dx  = (rho_for_flux[1:, :] - rho_for_flux[:-1, :]) / dx
    F[1:nx, :] = u0 * rho_face - nu_face * drho_dx

    # y-faces j+1/2
    rho_face = 0.5 * (rho_for_flux[:, 1:] + rho_for_flux[:, :-1])
    nu_face  = 0.5 * (nu[:, 1:] + nu[:, :-1])
    drho_dy  = (rho_for_flux[:, 1:] - rho_for_flux[:, :-1]) / dy
    G[:, 1:ny] = v0 * rho_face - nu_face * drho_dy

    # boundary faces (consistent with node boundary values)
    F[0, :]  = u0 * rho_for_flux[0, :]
    F[-1, :] = u0 * rho_for_flux[-1, :]
    G[:, 0]  = v0 * rho_for_flux[:, 0]
    G[:, -1] = v0 * rho_for_flux[:, -1]

    return F, G


def step_once_interpolated(rho_old: np.ndarray, rho_inter: np.ndarray, dt: float, nu: np.ndarray) -> np.ndarray:
    m = rho_old * V
    F, G = compute_fluxes_centered(rho_inter, nu)

    div_x = (F[1:nx+1, :] - F[0:nx, :]) * Ly
    div_y = (G[:, 1:ny+1] - G[:, 0:ny]) * Lx

    m_new = m - dt * (div_x + div_y)
    rho_new = m_new / V
    apply_bc(rho_new)
    return rho_new


def monotonicity_violations(rho_ref: np.ndarray, rho_new: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    viol = np.zeros_like(rho_new, dtype=bool)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            block = rho_ref[i - 1:i + 2, j - 1:j + 2]
            mn = block.min()
            mx = block.max()
            v = rho_new[i, j]
            if v < mn - tol or v > mx + tol:
                viol[i, j] = True
    viol[0, :] = viol[-1, :] = False
    viol[:, 0] = viol[:, -1] = False
    return viol


def step_with_aiv(
    rho_old: np.ndarray,
    dt: float,
    beta_limiter: float = 0.8,
    const_rho: float = 0.10,
    omega: float = 0.30,
    max_iters: int = 200,
    eps_rel: float = 1e-8,
    eps_abs: float = 1e-12,
):
    CFL = compute_cfl(dt)
    h = min(dx, dy)

    rho0 = rho_old.copy()
    apply_bc(rho0)

    # initial iterate rho^s
    rho_s = rho0.copy()

    # beta starts from CFL (depends on Courant number)
    beta = np.full_like(rho0, CFL)
    beta[0, :] = beta[-1, :] = beta[:, 0] = beta[:, -1] = 0.0

    for _ in range(max_iters):
        rho_inter = 0.5 * (rho_s + rho0)
        nu = beta * (h * h) / dt

        rho_new = step_once_interpolated(rho0, rho_inter, dt, nu)

        # check oscillations -> update beta
        viol = monotonicity_violations(rho0, rho_new)
        if np.any(viol):
            inc = const_rho * CFL
            idx = np.argwhere(viol)
            for i, j in idx:
                # update beta at node and 4-neighbors
                for di, dj in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        beta[ii, jj] = min(beta_limiter, beta[ii, jj] + inc)

        # relaxation update (convergence control)
        rho_s_next = omega * rho_new + (1.0 - omega) * rho_s
        apply_bc(rho_s_next)

        diff = float(np.max(np.abs(rho_s_next - rho_s)))
        thresh = float(eps_rel * np.max(np.abs(rho_s_next)) + eps_abs)

        rho_s = rho_s_next
        if (diff < thresh) and (not np.any(viol)):
            break

    return rho_s, beta


def run_sim(use_aiv: bool, T_end: float = 0.8, dt: float = 0.004, beta_limiter: float = 0.8):
    rho = np.where(X < 0.0, rho_L, rho_R).astype(float)
    apply_bc(rho)

    CFL = compute_cfl(dt)
    nsteps = int(np.ceil(T_end / dt))
    print(f"Run: T_end={T_end}, steps={nsteps}, CFL={CFL:.6f}, beta_limiter={beta_limiter}")

    beta_last = np.zeros_like(rho)
    for _ in range(nsteps):
        if use_aiv:
            rho, beta_last = step_with_aiv(rho, dt=dt, beta_limiter=beta_limiter)
        else:
            rho = step_once_interpolated(rho, rho, dt, nu=np.zeros_like(rho))

    return rho, beta_last

# Simulate
rho_no, _ = run_sim(use_aiv=False, T_end=0.8, dt=dt, beta_limiter=0.8)
rho_aiv, beta_last = run_sim(use_aiv=True,  T_end=0.8, dt=dt, beta_limiter=0.8)
print(f"No AIV: rho min/max = {rho_no.min():.6f} / {rho_no.max():.6f}")
print(f"With AIV: rho min/max = {rho_aiv.min():.6f} / {rho_aiv.max():.6f}; beta_last max={beta_last.max():.3f}")


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

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax3 = fig.add_subplot(111, projection="3d")
surf0 = ax3.plot_surface(X, Y, rho_no, cmap="viridis", linewidth=0, antialiased=True, vmin=vmin, vmax=vmax)
ax3.set_title("3D surface: ρ(x,y) — No AIV")
ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("ρ")
cbar3d0 = fig.colorbar(surf0, ax=ax3, shrink=0.6, pad=0.1)
cbar3d0.set_label("Плотность ρ")
plt.show()

fig = plt.figure(figsize=(10, 6))
ax3 = fig.add_subplot(111, projection="3d")
surf1 = ax3.plot_surface(X, Y, rho_aiv, cmap="viridis", linewidth=0, antialiased=True, vmin=vmin, vmax=vmax)
ax3.set_title("3D surface: ρ(x,y) — With AIV")
ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("ρ")
cbar3d1 = fig.colorbar(surf1, ax=ax3, shrink=0.6, pad=0.1)
cbar3d1.set_label("Плотность ρ")
plt.show()

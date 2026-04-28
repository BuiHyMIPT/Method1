import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
import time

# ============================================================
# Parameters
# ============================================================
gamma = 5.0 / 3.0   # or 1.0
# gamma = 1.0
cSound = 1.0
P_L = 2.0
P_R = 1.0
rho_R = 1.0

nx_cells, ny_cells = 80, 80
x_min, x_max = -1.0, 1.0
y_min, y_max = 0.0, 1.0



dt = 0.001
T_end = 0.30

# AIV parameters
beta_limiter = 0.2
beta_limiter_u = 0.3
const_rho = 0.005
const_u = 0.005

omega_time = 0.50
max_outer_iters = 20   # convergence of coupled iterate
max_inner_iters = 80   # monotonicity / beta-selection loop
eps_rel = 1e-10
eps_abs = 1e-12

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

# ============================================================
# Initial / boundary data
# ============================================================
rho_L = (P_L / cSound) ** (1.0 / gamma)
den = (1.0 / rho_R - 1.0 / rho_L)
if den <= 0.0:
    raise ValueError("Need rho_L > rho_R for the given formulas.")

front_aux = np.sqrt((P_L - P_R) / den)
D = front_aux / rho_R
u_L = D - front_aux / rho_L
u_R = 0.0
v_L = 0.0
v_R = 0.0

# ============================================================
# Grid / geometry
# ============================================================
nx, ny = nx_cells + 1, ny_cells + 1
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
dx = (x_max - x_min) / nx_cells
dy = (y_max - y_min) / ny_cells
X, Y = np.meshgrid(x, y, indexing="ij")

print(f"Cells grid = {nx_cells}x{ny_cells}")
# print(f"Nodes grid = {nx}x{ny}")
# print(f"dx = {dx:.6f}, dy = {dy:.6f}")


Lx = np.full((nx, ny), dx)
Ly = np.full((nx, ny), dy)
Lx[0, :] *= 0.5
Lx[-1, :] *= 0.5
Ly[:, 0] *= 0.5
Ly[:, -1] *= 0.5
V = Lx * Ly
h = min(dx, dy)

# ============================================================
# Helpers
# ============================================================
def pressure_from_rho(rho: np.ndarray) -> np.ndarray:
    return cSound * rho**gamma
    # return rho * cSound ** 2

def apply_bc(rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    rho[:, 0]  = rho[:, 1]
    rho[:, -1] = rho[:, -2]
    u[:, 0]    = u[:, 1]
    u[:, -1]   = u[:, -2]
    v[:, 0]    = v[:, 1]
    v[:, -1]   = v[:, -2]

    rho[0, :]  = rho_L
    rho[-1, :] = rho_R
    u[0, :]    = u_L
    u[-1, :]   = u_R
    v[0, :]    = v_L
    v[-1, :]   = v_R

def local_cfl_field(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return dt * (np.abs(u) / dx + np.abs(v) / dy)

def compute_cfl(dt: float, u: np.ndarray, v: np.ndarray) -> float:
    return float(np.max(local_cfl_field(u, v)))

def face_average_x(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a[1:, :] + a[:-1, :])

def face_average_y(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a[:, 1:] + a[:, :-1])

def node_to_xface(a: np.ndarray) -> np.ndarray:
    out = np.empty((nx + 1, ny))
    out[1:nx, :] = face_average_x(a)
    out[0, :] = a[0, :]
    out[-1, :] = a[-1, :]
    return out

def node_to_yface(a: np.ndarray) -> np.ndarray:
    out = np.empty((nx, ny + 1))
    out[:, 1:ny] = face_average_y(a)
    out[:, 0] = a[:, 0]
    out[:, -1] = a[:, -1]
    return out

def front_position(t: float) -> float:
    return D * t

def exact_fields(t: float):
    xf = front_position(t)
    rho_ex = np.where(X < xf, rho_L, rho_R).astype(float)
    u_ex   = np.where(X < xf, u_L,   u_R).astype(float)
    v_ex   = np.zeros_like(rho_ex)
    return rho_ex, u_ex, v_ex, xf

# ============================================================
# Fluxes and operators
# ============================================================
def compute_mass_fluxes(rho: np.ndarray, u: np.ndarray, v: np.ndarray, nu_rho: np.ndarray):
    F = np.zeros((nx + 1, ny))
    G = np.zeros((nx, ny + 1))

    F[1:nx, :] = face_average_x(rho) * face_average_x(u) \
               - face_average_x(nu_rho) * (rho[1:, :] - rho[:-1, :]) / dx

    G[:, 1:ny] = face_average_y(rho) * face_average_y(v) \
               - face_average_y(nu_rho) * (rho[:, 1:] - rho[:, :-1]) / dy

    F[0, :]  = rho[0, :]  * u[0, :]
    F[-1, :] = rho[-1, :] * u[-1, :]
    G[:, 0]  = rho[:, 0]  * v[:, 0]
    G[:, -1] = rho[:, -1] * v[:, -1]
    return F, G

def compute_div_integral(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    return (F[1:nx+1, :] - F[0:nx, :]) * Ly + (G[:, 1:ny+1] - G[:, 0:ny]) * Lx

def compute_div(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    return compute_div_integral(F, G) / V

def build_pressure_faces(rho: np.ndarray):
    """
    Thermodynamic pressure averaged to faces.
    """
    P_node = pressure_from_rho(rho)
    P_x = node_to_xface(P_node)
    P_y = node_to_yface(P_node)
    return P_x, P_y

def build_regularized_pressure_faces(rho: np.ndarray, u: np.ndarray, v: np.ndarray, nu_u: np.ndarray):
    """
    Regularized pressure on faces:
        \tilde\pi = P^(0.5) - \nu_u DIV(\rho u)
    """
    P_x, P_y = build_pressure_faces(rho)
    F0, G0 = compute_mass_fluxes(rho, u, v, np.zeros_like(rho))
    visc_corr_node = nu_u * compute_div(F0, G0)
    visc_x = node_to_xface(visc_corr_node)
    visc_y = node_to_yface(visc_corr_node)
    pi_x = P_x - visc_x
    pi_y = P_y - visc_y
    return pi_x, pi_y

# ============================================================
# Monotonicity / AIV
# ============================================================
def monotonicity_violations(ref: np.ndarray, cand: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    a = ref
    blocks = (
        a[:-2, :-2], a[1:-1, :-2], a[2:, :-2],
        a[:-2, 1:-1], a[1:-1, 1:-1], a[2:, 1:-1],
        a[:-2, 2:],  a[1:-1, 2:],  a[2:, 2:]
    )
    mn = np.minimum.reduce(blocks)
    mx = np.maximum.reduce(blocks)

    out = np.zeros_like(cand, dtype=bool)
    vv = cand[1:-1, 1:-1]
    out[1:-1, 1:-1] = (vv < mn - tol) | (vv > mx + tol)
    return out

def dilate_cross(mask: np.ndarray) -> np.ndarray:
    dil = mask.copy()
    dil[:-1, :] |= mask[1:, :]
    dil[1:, :]  |= mask[:-1, :]
    dil[:, :-1] |= mask[:, 1:]
    dil[:, 1:]  |= mask[:, :-1]
    return dil

def increase_beta_local(beta: np.ndarray, viol: np.ndarray, kr: np.ndarray, const_val: float, limiter: float):
    """
        beta_new = beta_old + const * k_r
    """
    if not np.any(viol):
        return beta, False

    area = dilate_cross(dilate_cross(viol))
    beta_new = beta.copy()
    beta_new[area] = np.minimum(limiter, beta_new[area] + const_val * kr[area])

    beta_new = np.minimum(limiter, np.maximum(beta_new, 0.0))

    updated = np.any(beta_new > beta + 1e-15)
    return beta_new, updated

# ============================================================
# One explicit time step without AIV
# ============================================================
def step_no_aiv(rho_old: np.ndarray, u_old: np.ndarray, v_old: np.ndarray):
    rho0 = rho_old.copy()
    u0 = u_old.copy()
    v0 = v_old.copy()
    apply_bc(rho0, u0, v0)

    # equation 1
    F, G = compute_mass_fluxes(rho0, u0, v0, np.zeros_like(rho0))
    rho_new = (rho0 * V - dt * compute_div_integral(F, G)) / V
    apply_bc(rho_new, u0, v0)

    # equation 2
    F, G = compute_mass_fluxes(rho_new, u0, v0, np.zeros_like(rho_new))
    P_x, P_y = build_pressure_faces(rho_new)

    press_x_int = (P_x[1:nx+1, :] - P_x[0:nx, :]) * Ly
    press_y_int = (P_y[:, 1:ny+1] - P_y[:, 0:ny]) * Lx

    u_x = node_to_xface(u0)
    v_x = node_to_xface(v0)
    u_y = node_to_yface(u0)
    v_y = node_to_yface(v0)

    conv_x_int = compute_div_integral(F * u_x, G * u_y)
    conv_y_int = compute_div_integral(F * v_x, G * v_y)

    m = rho_new * V
    mx = (rho0 * u0) * V
    my = (rho0 * v0) * V

    u_new = (mx - dt * (press_x_int + conv_x_int)) / m
    v_new = (my - dt * (press_y_int + conv_y_int)) / m
    apply_bc(rho_new, u_new, v_new)

    return rho_new, u_new, v_new

# ============================================================
# Joint AIV step
# ============================================================
def step_joint_with_aiv(rho_old: np.ndarray, u_old: np.ndarray, v_old: np.ndarray):
    rho0 = rho_old.copy()
    u0 = u_old.copy()
    v0 = v_old.copy()
    apply_bc(rho0, u0, v0)

    beta_rho = np.zeros_like(rho0)
    beta_u   = np.zeros_like(u0)

    rho_s = rho0.copy()
    u_s = u0.copy()
    v_s = v0.copy()

    beta_rho_last = beta_rho.copy()
    beta_u_last = beta_u.copy()

    for _s in range(max_outer_iters):
        rho_inter = omega_time * rho_s + (1.0 - omega_time) * rho0
        u_inter   = omega_time * u_s   + (1.0 - omega_time) * u0
        v_inter   = omega_time * v_s   + (1.0 - omega_time) * v0
        apply_bc(rho_inter, u_inter, v_inter)

        beta_rho_work = beta_rho.copy()
        beta_u_work   = beta_u.copy()

        rho_new = rho_s.copy()
        u_new = u_s.copy()
        v_new = v_s.copy()

        last_inner_viol = False

        for _k in range(max_inner_iters):
            kr = local_cfl_field(u_inter, v_inter)

            # ----- equation 1: continuity
            nu_rho = beta_rho_work * h * h / dt
            F_rho, G_rho = compute_mass_fluxes(rho_inter, u_inter, v_inter, nu_rho)
            rho_new = (rho0 * V - dt * compute_div_integral(F_rho, G_rho)) / V
            apply_bc(rho_new, u_inter, v_inter)

            # ----- equation 2: momentum
            F_m, G_m = compute_mass_fluxes(rho_new, u_inter, v_inter, nu_rho)

            nu_u = beta_u_work * h * h / dt
            pi_x, pi_y = build_regularized_pressure_faces(rho_new, u_inter, v_inter, nu_u)

            press_x_int = (pi_x[1:nx+1, :] - pi_x[0:nx, :]) * Ly
            press_y_int = (pi_y[:, 1:ny+1] - pi_y[:, 0:ny]) * Lx

            u_x = node_to_xface(u_inter)
            v_x = node_to_xface(v_inter)
            u_y = node_to_yface(u_inter)
            v_y = node_to_yface(v_inter)

            conv_x_int = compute_div_integral(F_m * u_x, G_m * u_y)
            conv_y_int = compute_div_integral(F_m * v_x, G_m * v_y)

            m = rho_new * V
            mx = (rho0 * u0) * V
            my = (rho0 * v0) * V

            u_new = (mx - dt * (press_x_int + conv_x_int)) / m
            v_new = (my - dt * (press_y_int + conv_y_int)) / m
            apply_bc(rho_new, u_new, v_new)

            viol_rho = monotonicity_violations(rho0, rho_new)
            viol_u   = monotonicity_violations(u0, u_new)
            viol_v   = monotonicity_violations(v0, v_new)
            viol_mom = viol_u | viol_v

            if (not np.any(viol_rho)) and (not np.any(viol_mom)):
                last_inner_viol = False
                break

            last_inner_viol = True
            updated_any = False

            if np.any(viol_rho):
                beta_rho_work, upd_rho = increase_beta_local(
                    beta_rho_work, viol_rho, kr, const_rho, beta_limiter
                )
                updated_any = updated_any or upd_rho

            if np.any(viol_mom):
                if not np.any(beta_u_work):
                    beta_u_seed = beta_rho_work.copy()
                else:
                    beta_u_seed = beta_u_work.copy()

                beta_u_work, upd_u = increase_beta_local(
                    beta_u_seed, viol_mom, kr, const_u, beta_limiter_u
                )
                updated_any = updated_any or upd_u

            if not updated_any:
                break

        # accept the beta values from the inner loop
        beta_rho = beta_rho_work.copy()
        beta_u = beta_u_work.copy()
        beta_rho_last = beta_rho.copy()
        beta_u_last = beta_u.copy()

        # convergence of the WHOLE coupled iterate
        diff = max(
            float(np.max(np.abs(rho_new - rho_s))),
            float(np.max(np.abs(u_new - u_s))),
            float(np.max(np.abs(v_new - v_s))),
        )
        scale = max(
            float(np.max(np.abs(rho_new))),
            float(np.max(np.abs(u_new))),
            float(np.max(np.abs(v_new))),
            1.0
        )
        thresh = eps_rel * scale + eps_abs

        rho_s = rho_new
        u_s = u_new
        v_s = v_new

        if diff < thresh and (not last_inner_viol):
            break

    return rho_s, u_s, v_s, beta_rho_last, beta_u_last

# ============================================================
# Simulation
# ============================================================
def initial_fields():
    rho = np.where(X < 0.0, rho_L, rho_R).astype(float)
    u = np.where(X < 0.0, u_L, u_R).astype(float)
    v = np.zeros_like(rho)
    apply_bc(rho, u, v)
    return rho, u, v

def run_sim(use_aiv: bool):
    rho, u, v = initial_fields()
    nsteps = int(np.ceil(T_end / dt))

    beta_rho_last = np.zeros_like(rho)
    beta_u_last = np.zeros_like(rho)

    for _ in range(nsteps):
        if use_aiv:
            rho, u, v, beta_rho_last, beta_u_last = step_joint_with_aiv(rho, u, v)
        else:
            rho, u, v = step_no_aiv(rho, u, v)

    return rho, u, v, beta_rho_last, beta_u_last

# ============================================================
# Plotting
# ============================================================
def plot_2d3d(field_no, field_aiv, field_exact, title_main, short_label, out_png, xf):
    vmin = float(min(field_no.min(), field_aiv.min(), field_exact.min()))
    vmax = float(max(field_no.max(), field_aiv.max(), field_exact.max()))

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0], projection="3d")
    ax11 = fig.add_subplot(gs[1, 1], projection="3d")
    ax12 = fig.add_subplot(gs[1, 2], projection="3d")

    for ax, field, title in (
        (ax00, field_no, "No AIV"),
        (ax01, field_aiv, "With AIV"),
        (ax02, field_exact, "Exact"),
    ):
        im = ax.imshow(field.T, origin="lower", extent=[x_min, x_max, y_min, y_max],
                       aspect="auto", vmin=vmin, vmax=vmax)
        ax.axvline(x=xf, color="w", linestyle="--", linewidth=2, label=f"x = D t = {xf:.3f}")
        ax.set_title(f"{title}: {title_main}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax, field, title in (
        (ax10, field_no, "No AIV"),
        (ax11, field_aiv, "With AIV"),
        (ax12, field_exact, "Exact"),
    ):
        surf = ax.plot_surface(X, Y, field, cmap="viridis", linewidth=0, antialiased=True,
                               vmin=vmin, vmax=vmax)
        ax.set_title(f"3D: {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(short_label)
        fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_main}; grid {nx_cells}x{ny_cells}; gamma={gamma}; front x=Dt={xf:.3f}", fontsize=14)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_centerline_compare(x, num_no, num_aiv, exact, ylabel, out_png, xf):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(x, num_no, label="No AIV", linewidth=2)
    ax.plot(x, num_aiv, label="With AIV", linewidth=2)
    ax.plot(x, exact, "--", label="Exact", linewidth=2)
    ax.axvline(x=xf, color="k", linestyle=":", linewidth=1.5, label=f"x = D t = {xf:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Centerline comparison for {ylabel}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================
def main():
    t0 = time.perf_counter()
    rho_no, u_no, v_no, _, _ = run_sim(use_aiv=False)
    t1 = time.perf_counter()
    print(f"No AIV time: {t1 - t0:.6f} s")

    t2 = time.perf_counter()
    rho_aiv, u_aiv, v_aiv, beta_rho_last, beta_u_last = run_sim(use_aiv=True)
    t3 = time.perf_counter()
    print(f"With AIV time: {t3 - t2:.6f} s")
    print(f"Время выполнения (total): {t3 - t0:.6f} s")

    rho_ex, u_ex, v_ex, xf = exact_fields(T_end)
    jmid = ny // 2

    print(f"D = {D:.6f}, front position x = D*T_end = {xf:.6f}")
    print(f"No AIV : rho min/max = {rho_no.min():.6f} / {rho_no.max():.6f}; "
          f"u min/max = {u_no.min():.6f} / {u_no.max():.6f}; "
          f"CFL={compute_cfl(dt, u_no, v_no):.6f}")

    print(f"With AIV: rho min/max = {rho_aiv.min():.6f} / {rho_aiv.max():.6f}; "
          f"u min/max = {u_aiv.min():.6f} / {u_aiv.max():.6f}; "
          f"beta_rho_max={beta_rho_last.max():.6f}; beta_u_max={beta_u_last.max():.6f}; "
          f"CFL={compute_cfl(dt, u_aiv, v_aiv):.6f}")

    plot_2d3d(rho_no, rho_aiv, rho_ex, "Density field rho(x,y)",
              "rho", OUTDIR / "rho_with_exact_front.png", xf)
    plot_2d3d(u_no, u_aiv, u_ex, "Velocity field u(x,y)",
              "u", OUTDIR / "u_with_exact_front.png", xf)

    plot_centerline_compare(x, rho_no[:, jmid], rho_aiv[:, jmid], rho_ex[:, jmid],
                            "rho", OUTDIR / "rho_centerline.png", xf)
    plot_centerline_compare(x, u_no[:, jmid], u_aiv[:, jmid], u_ex[:, jmid],
                            "u", OUTDIR / "u_centerline.png", xf)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im1 = ax[0].imshow(beta_rho_last.T, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    ax[0].set_title("beta_rho")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

    im2 = ax[1].imshow(beta_u_last.T, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    ax[1].set_title("beta_u")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    plt.savefig(OUTDIR / "beta_with_exact_front.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved outputs to: {OUTDIR.resolve()}")

if __name__ == '__main__':
    main()

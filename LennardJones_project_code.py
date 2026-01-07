import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import scipy.spatial.distance as spd
from scipy.spatial import cKDTree
import time

# ---------- MODE SELECTION ----------
TEST_MODE = True # Set to False for full simulation

# Test mode solely exists to run a quick simulation for debugging purposes.
if TEST_MODE:
    print("="*60, flush=True)
    print("RUNNING IN TEST MODE (N=128, 1k eq steps, 5k prod steps, ~1.5min max)")
    print("="*60, flush=True)
    N = 128
    steps_eq = 500
    steps_prod = 1000
    save_every = 50
    max_runtime_seconds = 120  # ~2 minutes
else:
    print("="*60, flush=True)
    print("RUNNING IN FULL MODE (N=512, 2k eq steps, 20k prod steps, ~30min max)")
    print("="*60, flush=True)
    N = 512
    steps_eq = 2000
    steps_prod = 10000
    max_runtime_seconds = 2200  # ~30 minutes

# ---------- PARAMETERS ----------
m, sigma = 1.0, 1.0
rho = 0.6
L = np.sqrt(N / rho)
x_min, x_max = -L/2, L/2
y_min, y_max = -L/2,L/2  


y_min, y_max = -L/2, L/2

dt = 0.001
cutoff_radius = 3.0 * sigma
cluster_rc = 1.5 * sigma
T_target = 0.8
save_every = 100
progress_interval = 500

epsilon_vals = [1, 2, 3, 4, 5]

# ---------- FUNCTIONS ----------
def init_config():
    x = np.random.uniform(x_min, x_max, N)
    y = np.random.uniform(y_min, y_max, N)
    v_scale = np.sqrt(2 * T_target)
    vx = np.random.normal(0, v_scale, N)
    vy = np.random.normal(0, v_scale, N)
    return x, y, vx, vy

def list_neighbours(x, y):
    points = np.c_[x, y]
    tree = cKDTree(points)
    indices = tree.query_ball_point(points, r=cutoff_radius)
    neighbours = [np.array([j for j in idx if j != i], dtype=int) for i, idx in enumerate(indices)]
    return neighbours

def total_force(x, y, eps_val, neighbours):
    Fx, Fy = np.zeros(N), np.zeros(N)
    for i in range(N):
        if len(neighbours[i]) == 0:
            continue
        drx = x[i] - x[neighbours[i]]
        dry = y[i] - y[neighbours[i]]
        r2 = drx**2 + dry**2
        r2 = np.maximum(r2, 0.01)
        ka2 = sigma**2 / r2
        F = np.clip(24 * eps_val / np.sqrt(r2) * (2*ka2**6 - ka2**3), -1e6, 1e6)
        Fx[i] = np.sum(F * drx / np.sqrt(r2))
        Fy[i] = np.sum(F * dry / np.sqrt(r2))
    return Fx, Fy

def cluster_ids(x, y):
    dist = spd.cdist(np.c_[x, y], np.c_[x, y])
    adj = (dist < cluster_rc) & (dist > 0)
    n_clust, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels, minlength=n_clust)
    return n_clust, sizes.mean(), sizes.max()

def anderson_thermostat(vx, vy):
    v_scale = np.sqrt(2 * T_target)
    vx[:] = np.random.normal(0, v_scale, N)
    vy[:] = np.random.normal(0, v_scale, N)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

# ---------- MAIN SIMULATION ----------
msd_plateau = []
avg_cluster = []
time_limit_hit = False
overall_start_time = time.time()

print(f"SIMULATION STARTED at {time.strftime('%H:%M:%S')}", flush=True)
print(f"Time limit: {max_runtime_seconds} seconds", flush=True)
print("-"*60, flush=True)

for eps_idx, eps in enumerate(epsilon_vals):
    if time.time() - overall_start_time > max_runtime_seconds:
        print(f"TIME LIMIT HIT before starting epsilon = {eps}", flush=True)
        break
    
    print(f"\n{'='*60}", flush=True)
    print(f"STARTING epsilon = {eps} ({eps_idx+1}/{len(epsilon_vals)})", flush=True)
    print(f"{'='*60}", flush=True)
    
    x, y, vx, vy = init_config()
    x0, y0 = x.copy(), y.copy()
    neighbours = list_neighbours(x, y)
    
    # EQUILIBRATION
    for step in range(steps_eq):
        x_half = x + 0.5*vx*dt
        y_half = y + 0.5*vy*dt
        fx, fy = total_force(x_half, y_half, eps, neighbours)
        vx += fx/m*dt
        vy += fy/m*dt
        x = x_half + 0.5*vx*dt
        y = y_half + 0.5*vy*dt
        
        for i in range(N):
            if x[i] < x_min or x[i] > x_max: vx[i] *= -1; x[i] = np.clip(x[i], x_min, x_max)
            if y[i] < y_min or y[i] > y_max: vy[i] *= -1; y[i] = np.clip(y[i], y_min, y_max)
        
        if step % 100 == 0:
            anderson_thermostat(vx, vy)
        if step % 10 == 0:
            neighbours = list_neighbours(x, y)
    
    # PRODUCTION
    msd_series = []
    clust_series = []
    for step in range(steps_prod):
        x_half = x + 0.5*vx*dt
        y_half = y + 0.5*vy*dt
        fx, fy = total_force(x_half, y_half, eps, neighbours)
        vx += fx/m*dt
        vy += fy/m*dt
        x = x_half + 0.5*vx*dt
        y = y_half + 0.5*vy*dt
        
        for i in range(N):
            if x[i] < x_min or x[i] > x_max: vx[i] *= -1; x[i] = np.clip(x[i], x_min, x_max)
            if y[i] < y_min or y[i] > y_max: vy[i] *= -1; y[i] = np.clip(y[i], y_min, y_max)
        
        if step % 100 == 0:
            anderson_thermostat(vx, vy)
        if step % 10 == 0:
            neighbours = list_neighbours(x, y)
        
        if step % save_every == 0:
            msd_series.append(np.mean((x-x0)**2 + (y-y0)**2))
            n_clust, avg_S, max_S = cluster_ids(x, y)
            clust_series.append(avg_S)
            
            if time.time() - overall_start_time > max_runtime_seconds:
                print(f"TIME LIMIT HIT at step {step}", flush=True)
                time_limit_hit = True
                break
    
    if time_limit_hit:
        break
    
    if len(msd_series) > 0:
        msd_plateau.append(np.mean(msd_series[-10:]))
        avg_cluster.append(np.mean(clust_series[-10:]))

# ---------- PLOTTING ----------
if len(msd_plateau) >= 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(epsilon_vals[:len(msd_plateau)], msd_plateau, 'o-', color='#0077BB', lw=2.5, markersize=8)
    ax1.set_xlabel('LJ interaction strength epsilon'); ax1.set_ylabel('MSD plateau')
    ax1.set_title('Particle Mobility'); ax1.grid(alpha=0.3)

    ax2.plot(epsilon_vals[:len(avg_cluster)], avg_cluster, 'o-', color='#EE7733', lw=2.5, markersize=8)
    ax2.set_xlabel('LJ interaction strength epsilon'); ax2.set_ylabel('Average cluster size')
    ax2.set_title('Cluster Growth'); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('poster_image_A.pdf')
    plt.show()
else:
    print("Not enough data to plot.")

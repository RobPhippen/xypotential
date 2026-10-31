
"""
2D XY Model
============

A simple Monte Carlo simulation of the two-dimensional XY model.

Each lattice site contains a planar spin with orientation theta.

Interaction:
    E_ij = -J cos(theta_i - theta_j)

Boundary conditions:
    Periodic

Monte Carlo:
    Shuffled sequential Metropolis updates
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Interaction energy
# ============================================================

def pair_energy(theta1, theta2, J=1.0):
    """
    Interaction energy between two neighbouring spins.

    Parameters
    ----------
    theta1, theta2 : float or ndarray
        Spin orientations in radians. If arrays are supplied, NumPy applies
        the formula elementwise, returning one energy per corresponding pair.
    J : float
        Coupling constant.

    Returns
    -------
    float or ndarray
        Pair interaction energy.
    """

    return -J * np.cos(theta1 - theta2)


# ============================================================
# 2. Total energy
# ============================================================

def total_energy(theta, J=1.0):
    """
    Calculate the total energy of the lattice.

    Each nearest-neighbour bond is counted once.
    Periodic boundary conditions are used.
    """

    # np.roll shifts the array and wraps values across the edges. This gives
    # each site's right and downward neighbours with periodic boundaries:
    # right[i, j] is theta[i, (j + 1) % n], and
    # down[i, j] is theta[(i + 1) % n, j].
    right = np.roll(theta, -1, axis=1)
    down = np.roll(theta, -1, axis=0)

    # theta, right, and down are arrays, so pair_energy evaluates all
    # corresponding bonds elementwise; summing gives the lattice energy.
    energy = np.sum(
        pair_energy(theta, right, J)
        + pair_energy(theta, down, J)
    )

    return energy


# ============================================================
# 3. Magnetisation
# ============================================================

def order_parameter(theta):
    """
    Calculate the XY magnetisation.

    m = |mean(exp(i theta))|

    Returns a value between 0 and 1.
    """

    mx = np.mean(np.cos(theta))
    my = np.mean(np.sin(theta))

    return np.sqrt(mx**2 + my**2)


# ============================================================
# 4. Monte Carlo sweep
# ============================================================

def mc_step(theta, T, J=1.0, scale=None):
    """
    Perform one Monte Carlo sweep.

    Every lattice site is visited exactly once,
    in a randomly shuffled order.

    Accepted changes are applied immediately.

    Parameters
    ----------
    theta : ndarray
        Spin orientation array.
    T : float
        Temperature.
    J : float
        Coupling constant.
    scale : float or None
        Standard deviation of the proposed angle change. Defaults to 0.5 radians.

    Returns
    -------
    float
        Fraction of proposed updates accepted.
    """

    n = theta.shape[0]

    if theta.shape != (n, n):
        raise ValueError("Expected a square lattice.")

    if T <= 0:
        raise ValueError("Temperature must be positive.")

    if scale is None:
        scale = 0.5

    # Visit every XY spin once, in a randomized site order.
    sites = np.random.permutation(n * n)

    accepted = 0

    for site in sites:

        # Convert linear site index to lattice coordinates
        ix, iy = divmod(site, n)

        # Read the four periodic nearest-neighbour XY angles.
        neighbours = (
            theta[(ix + 1) % n, iy],
            theta[(ix - 1) % n, iy],
            theta[ix, (iy + 1) % n],
            theta[ix, (iy - 1) % n],
        )

        # Current orientation
        old = theta[ix, iy]

        # Propose a new orientation
        new = old + np.random.normal(scale=scale)

        # Sum the bond-energy changes caused by this spin proposal.
        dE = 0.0

        for neighbour in neighbours:

            dE += (
                pair_energy(new, neighbour, J)
                - pair_energy(old, neighbour, J)
            )

        # Apply the XY Metropolis criterion to the proposed energy change.
        if dE <= 0.0:

            theta[ix, iy] = new
            accepted += 1

        elif np.random.random() < np.exp(-dE / T):

            theta[ix, iy] = new
            accepted += 1

    return accepted / (n * n)


# ============================================================
# 5. Simulation
# ============================================================

def simulate(
    n=20,
    T=0.5,
    steps=1000,
    J=1.0,
    seed=1234,
):
    """
    Run the XY model simulation.

    Parameters
    ----------
    n : int
        Linear lattice size.
    T : float
        Temperature.
    steps : int
        Number of Monte Carlo sweeps.
    J : float
        Coupling constant.
    seed : int
        Random seed.

    Returns
    -------
    theta : ndarray
        Final spin configuration.
    energies : ndarray
        Energy per site at each sweep.
    order : ndarray
        Magnetisation at each sweep.
    acceptance : ndarray
        Acceptance fraction at each sweep.
    """

    if n < 2:
        raise ValueError("Lattice size must be at least 2.")

    if T <= 0:
        raise ValueError("Temperature must be positive.")

    if steps < 1:
        raise ValueError("Number of steps must be positive.")

    np.random.seed(seed)

    # Draw independent planar-spin angles over the full circle.
    theta = np.random.uniform(
        0.0,
        2.0 * np.pi,
        size=(n, n)
    )

    energies = np.empty(steps)
    order = np.empty(steps)
    acceptance = np.empty(steps)

    # Monte Carlo simulation
    for step in range(steps):

        acceptance[step] = mc_step(theta, T, J)

        # Store the XY bond energy normalized by the number of sites.
        energies[step] = total_energy(theta, J) / (n * n)

        # Track planar magnetisation from the mean spin-vector components.
        order[step] = order_parameter(theta)

    return theta, energies, order, acceptance


# ============================================================
# 6. Plotting
# ============================================================



def plot_results(theta, energies, order, acceptance):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Final spin configuration as arrows
    n = theta.shape[0]
    x, y = np.meshgrid(np.arange(n), np.arange(n))

    u = np.cos(theta)
    v = np.sin(theta)

    axes[0, 0].quiver(
        x, y, u, v,
        angles="xy",
        scale_units="xy",
        scale=1.5,
        pivot="middle",
    )
    axes[0, 0].set_title("Final XY spin configuration")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlim(-0.5, n - 0.5)
    axes[0, 0].set_ylim(-0.5, n - 0.5)

    # Energy per site
    axes[0, 1].plot(energies)
    axes[0, 1].set_title("Energy per site")
    axes[0, 1].set_xlabel("Monte Carlo sweep")
    axes[0, 1].set_ylabel("Energy")

    # Order parameter
    axes[1, 0].plot(order)
    axes[1, 0].set_title("XY order parameter")
    axes[1, 0].set_xlabel("Monte Carlo sweep")
    axes[1, 0].set_ylabel("Magnetisation magnitude")

    # Acceptance rate
    axes[1, 1].plot(acceptance)
    axes[1, 1].set_title("Acceptance rate")
    axes[1, 1].set_xlabel("Monte Carlo sweep")
    axes[1, 1].set_ylabel("Fraction accepted")

    plt.tight_layout()
    plt.show()

# ============================================================
# 7. Main program
# ============================================================

if __name__ == "__main__":

    # Simulation parameters
    n = 40
    T = 0.5
    steps = 1000
    J = 1.0

    # Run simulation
    theta, energies, order, acceptance = simulate(
        n=n,
        T=T,
        steps=steps,
        J=J,
        seed=1234,
    )

    # Print summary
    print("2D XY model")
    print("------------")
    print(f"Lattice size: {n} x {n}")
    print(f"Temperature: {T}")
    print(f"Coupling: {J}")
    print(f"Monte Carlo sweeps: {steps}")
    print()
    print(f"Final energy per site: {energies[-1]:.6f}")
    print(f"Final magnetisation: {order[-1]:.6f}")
    print(f"Final acceptance rate: {acceptance[-1]:.4f}")

    # Plot results
    plot_results(theta, energies, order, acceptance)

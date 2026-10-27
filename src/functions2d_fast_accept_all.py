"""Faster variant of `choosemove` from `functions2d.py` (Algorithm 1 of Altmann & Spreer).

The original `choosemove` enumerates every distinct neighbour (up to isomorphism)
of the current triangulation at each step, which requires computing an
isomorphism signature for every candidate move. This version instead picks a
single valid site (triangle, edge or vertex) uniformly at random, performs the move, and
corrects for the resulting bias with a Metropolis-Hastings step using the ratio
of automorphism group sizes. Only one isomorphism signature is computed per step.

The up/down/flip proposal is the same as the original: an up move (1-3) is proposed
with probability exp(-gamma * f_2), otherwise a down move (3-1) and flip (2-2) proposal is proposed with probability (1-exp(-gamma * f_2))/2.
"""

import math
import random

from regina.engine import Triangulation2


def get_aut(tri: Triangulation2) -> int:
    """Return the size of the automorphism group of a triangulation.

    Parameters
    ----------
    tri : Triangulation2
        The triangulation.

    Returns
    -------
    int
        The number of combinatorial automorphisms of `tri`.
    """
    return len(tri.findAllIsomorphisms(tri))


def get_f_vector(tri: Triangulation2) -> list[int]:
    """Return the f-vector of a triangulation.

    Parameters
    ----------
    tri : Triangulation2
        The triangulation.

    Returns
    -------
    list[int]
        The f-vector (f_0, f_1, f_2).
    """
    f = tri.fVector()
    return f


def perform_move(tri: Triangulation2, a: int, i: int) -> Triangulation2:
    """Perform a Pachner move at a given site, without checking validity.

    The input triangulation is copied and left unchanged.

    Parameters
    ----------
    tri : Triangulation2
        The initial triangulation.
    a : int
        The move type. 0 is a 1-3 move (up, on triangle `i`), 1 is a 2-2 move (flip, on edge `i`), 2 is a 3-1 move (down, on vertex `i`).
    i : int
        The index of the triangle (a=0) or edge (a=1) or vertex (a=2) to move on. Should come from `get_sites`.

    Returns
    -------
    Triangulation2
        The modified copy of `tri`.
    """
    target = Triangulation2(tri)

    if a == 0:
        target.pachner(target.triangle(i), False, True)
    if a == 1:
        target.pachner(target.edge(i), False, True)
    if a == 2:
        target.pachner(target.vertex(i), False, True)

    return target


def get_sites(tri: Triangulation2, f: list[int], a: int) -> list[int]:
    """List all sites at which a given move type can be performed.

    Parameters
    ----------
    tri : Triangulation2
        The triangulation.
    f : list[int]
        The f-vector of `tri`.
    a : int
        The move type, see `perform_move`.

    Returns
    -------
    list[int]
        Indices of the triangles (a=0) or edges (a=1) or vertices (a=2) at which the move is valid.
    """
    sites = []
    target = Triangulation2(tri)

    if a == 0:
        for t in range(f[2]):
            if target.pachner(target.triangle(t), True, False):
                sites.append(t)

    if a == 1:
        for e in range(f[1]):
            if target.pachner(target.edge(e), True, False):
                sites.append(e)

    if a == 2:
        for v in range(f[0]):
            if target.pachner(target.vertex(v), True, False):
                sites.append(v)

    return sites


### Choose move from proposal
def choosemove(
    iso: str,
    gamma: float,
    f: list[int] | None = None,
    aut: int | None = None,
) -> tuple[str, list[int], int]:
    """Perform one step of the MCMC chain.

    Proposes an up move with probability exp(-gamma * f_2) (else a down or flip move),
    stays put with the same probability as the original `choosemove`, and
    otherwise moves at a uniformly random valid site. The move is accepted with
    probability min(1, |Aut(T')| / |Aut(T)|).

    Passing `f` and `aut` from the previous step avoids recomputing them.

    Parameters
    ----------
    iso : str
        Isomorphism signature (`isoSig_RidgeDegrees`) of the current triangulation.
    gamma : float
        Parameter controlling the up/down/flip proposal, and hence the typical number of triangles.
    f : list[int] | None, optional
        f-vector of the current triangulation. Computed if None.
    aut : int | None, optional
        Automorphism group size of the current triangulation. Computed if None.

    Returns
    -------
    tuple[str, list[int], int]
        The isomorphism signature, f-vector and automorphism group size of the next state.

    Raises
    ------
    RuntimeError
        If no branch returns (should never happen).
    """
    tri = Triangulation2.fromIsoSig(iso)

    if aut is None:
        aut = get_aut(tri)
    if f is None:
        f = get_f_vector(tri)

    # This is alpha() from functions2d.py
    x = random.random()
    p_up = math.exp((-1) * gamma * f[2])
    if x < p_up:
        a = 0
    elif x < (1 + p_up) / 2:
        a = 1
    else:
        a = 2

    sites = get_sites(tri, f, a)
    num_sites = len(sites)
    i = random.random()
    u = random.random()

    # go up (1-3)
    if a == 0:
        # stay where you are
        if i > float(num_sites) / float(f[2]): # divide by number of possible triangles to act on 
            return iso, f, aut
        # go up (very likely)
        else:
            tri_prop = perform_move(tri, a, random.choice(sites))
            aut_prop = get_aut(tri_prop)
            accept_prob = min(1, aut_prop / aut)
            if u > accept_prob:
                return iso, f, aut
            return (
                tri_prop.isoSig_RidgeDegrees(),
                [
                    f[0] + 1,
                    f[1] + 3,
                    f[2] + 2,
                ],
                aut_prop,
            )

    # edge flip (2-2)
    if a == 1:
        # stay where you are
        if i > float(num_sites) / float(f[1]): 
            return iso, f, aut
        # flip (very likely)
        else:
            tri_prop = perform_move(tri, a, random.choice(sites))
            aut_prop = get_aut(tri_prop)
            accept_prob = min(1, aut_prop / aut)
            if u > accept_prob: 
                return iso, f, aut
            return (
                tri_prop.isoSig_RidgeDegrees(), f, aut_prop,
            )

    # go down (3-1)
    elif a == 2:
        # stay where you are
        # avoid division by zero
        if f[2] <= 2:
            return iso, f, aut
        if i > float(num_sites) / float(f[2] - 2):
            return iso, f, aut
        # go down (unlikely)
        else:
            tri_prop = perform_move(tri, a, random.choice(sites))
            aut_prop = get_aut(tri_prop)
            accept_prob = min(1, aut_prop / aut)
            if u > accept_prob:
                return iso, f, aut
            return (
                tri_prop.isoSig_RidgeDegrees(),
                [
                    f[0] - 1,
                    f[1] - 3,
                    f[2] - 2,
                ],
                aut_prop,
            )

    raise RuntimeError("Should not reach this point in choosemove")


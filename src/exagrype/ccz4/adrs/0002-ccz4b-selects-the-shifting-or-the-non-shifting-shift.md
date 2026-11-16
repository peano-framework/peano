# ADR 0002: CCZ4b selects the shifting or the non-shifting shift

Status: proposed
Date: 2026-11-16

## Context

The Gamma driver comes in variants which differ in the advection terms of its equations.
van Meter et al. (arXiv:gr-qc/0605030, Table 1) list the eight combinations of three such
terms. Two of them evolve moving punctures cleanly: the shifting shift with all three terms
(their Eqs. 21-22, Case 8) and the non-shifting shift without any (Eqs. 23-24, Case 1). The
ExaGRyPE paper (arXiv:2406.11626, Section 2.1) states that both are available.

The kernels had a switch for this, `CCZ4bs`, now `CCZ4b`. With 1, they solve the shifting
shift, as all setups with an evolving shift do. With 0, they did not solve the non-shifting
shift:

- The equation of b holds the time derivative of Ghat. The kernels wrote it as "time
  derivative of Ghat plus bs times (advection of b minus advection of Ghat)", which is
  right as long as that time derivative comprises the advection of Ghat. Since the
  advection terms have moved out of the non-conservative product (the numerical scheme
  adds them with lopsided stencils, and the 2022 kernels already multiplied them by zero),
  it does not. With 0, the equation hence lacked the advection of Ghat, which is Case 3
  of the table.
- The equation of the derivatives of the shift, `B_j^i`, kept the product `B_k^i B_j^k`.
  The equations of the secondary entries are the derivatives of the equations of the
  primary ones (Dumbser et al., arXiv:1707.09910, below their Eq. 10), and this product is the
  derivative of the advection term `beta^k B_k^i` of the shift equation. Without that term,
  it has to vanish.

The Fortran code which the kernels were ported from (`examples/exahype2/ccz4/PDE.f90`,
routine `PDEFusedSrcNCP`, in the history of the repository since commit 5c1efc288) has
both right: its time derivative of Ghat holds the advection of Ghat, and it writes
`dtB = dtB + bs*( beta(1)*dBB(1,:,:) + ... + MATMUL(BB,BB) )`. The C++ port split this
routine into a source and a non-conservative product. The product `MATMUL(BB,BB)` went into
the source and kept the factor `sk` only, and the advection of Ghat was multiplied by zero.

No setup ever used an evolving shift with 0, which is why this went unnoticed.

## Decision

With s for `CCZ4s` and b for `CCZ4b`, the Gamma driver reads

    d_t beta^i = s ( b beta^k B_k^i + f b^i )
    d_t b^i    = s ( d_t Ghat^i - eta b^i + b ( beta^k d_k b^i - beta^k d_k Ghat^i ) )
    d_t B_j^i  = s ( b ( beta^k d_k B_j^i + B_k^i B_j^k ) + f d_j b^i ) + terms with mu

where `d_t Ghat^i` is the complete right-hand side of the equation of Ghat. b=1 is the
shifting shift, b=0 the non-shifting shift, and the third equation is the derivative of the
first one for both. In the code:

- The source multiplies `B_k^i B_j^k` with `CCZ4b`.
- The non-conservative product adds `beta^k d_k Ghat^i` to the equation of b if `CCZ4b` is 0
  and the advection terms are not compiled into the kernel.
- `lopsided_advection_quantities()` of the solvers stays as it is: The numerical scheme
  advects b and the derivatives of the shift (in the second-order formulation: the shift
  and b) only if both switches are 1.

## Consequences

- For the non-shifting shift, the advection of Ghat within the equation of b is computed
  from the derivatives that the non-conservative product receives, i.e. with central
  differences, whereas the scheme advects Ghat itself with lopsided stencils. The two differ
  by the truncation error. To use the same stencil, the scheme would have to add the
  advection of one quantity to the right-hand side of another one, which
  `exahype2::fd::addLopsidedAdvection()` cannot do.
- The Finite Volume schemes add no advection terms at all. With them, the equation of b of
  the non-shifting shift holds an advection of Ghat which the equation of Ghat lacks. An
  evolving shift is not solved correctly with Finite Volumes in either variant.
- `exagrype::ccz4::tests::GammaDriverTest` checks that the kernels of the two variants
  differ in exactly the terms above. The non-shifting shift has not been run on a black
  hole yet.
- The default of `CCZ4b` is 1, i.e. whoever switches on the shift with `CCZ4s=1` obtains
  the shifting shift, which the paper calls the default gauge. The default used to be 0.
  As the default of `CCZ4s` is 0, the default setup is the same as before.

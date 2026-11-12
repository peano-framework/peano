import exahype2

def set_euler_pde_terms(solver, dimensions: int, stateless: bool):
    if dimensions == 2:
        solver.add_pde_term(
            exahype2.api.Flux(
                implementation="""
            const double irho  = 1.0 / Q[0];
            constexpr double gamma = 1.4;
            const double p = (gamma - 1.0) * (Q[3] - 0.5 * irho * (Q[1] * Q[1] + Q[2] * Q[2]));

            switch (normal) {
                case 0:
                F[0] = Q[1];
                F[1] = irho * Q[1] * Q[1] + p;
                F[2] = irho * Q[1] * Q[2];
                F[3] = irho * Q[1] * (Q[3] + p);
                break;
                case 1:
                F[0] = Q[2];
                F[1] = irho * Q[2] * Q[1];
                F[2] = irho * Q[2] * Q[2] + p;
                F[3] = irho * Q[2] * (Q[3] + p);
                break;
            }
            """,
                stateless=stateless,
            )
        )
        solver.add_pde_term(
            exahype2.api.MaxEigenvalue(
                implementation="""
            // Eigenvalues of the Euler flux Jacobian in direction "normal" are
            // u_n +/- c, where u_n is the *velocity* component (momentum/density),
            // not the raw momentum component - several earlier ExaHyPE
            // applications/benchmarks used the raw momentum here by mistake, which
            // silently gives a wrong (and usually too small) CFL time step bound.
            const double irho  = 1.0 / Q[0];
            constexpr double gamma = 1.4;
            const double p = (gamma - 1.0) * (Q[3] - 0.5 * irho * (Q[1] * Q[1] + Q[2] * Q[2]));
            const double c = gamma * p * irho;

            switch (normal) {
                case 0:
                lambdaMax[0] = std::fmax(std::fabs(Q[1] * irho + c), std::fabs(Q[1] * irho - c));
                break;
                case 1:
                lambdaMax[0] = std::fmax(std::fabs(Q[2] * irho + c), std::fabs(Q[2] * irho - c));
                break;
            }
            """,
                stateless=stateless,
            )
        )
    else:
        solver.add_pde_term(
            exahype2.api.Flux(
                implementation="""
            const double irho  = 1.0 / Q[0];
            constexpr double gamma = 1.4;
            const double p = (gamma - 1.0) * (Q[4] - 0.5 * irho * (Q[1] * Q[1] + Q[2] * Q[2] + Q[3] * Q[3]));

            switch (normal) {
                case 0:
                F[0] = Q[1];
                F[1] = irho * Q[1] * Q[1] + p;
                F[2] = irho * Q[1] * Q[2];
                F[3] = irho * Q[1] * Q[3];
                F[4] = irho * Q[1] * (Q[4] + p);
                break;
                case 1:
                F[0] = Q[2];
                F[1] = irho * Q[2] * Q[1];
                F[2] = irho * Q[2] * Q[2] + p;
                F[3] = irho * Q[2] * Q[3];
                F[4] = irho * Q[2] * (Q[4] + p);
                break;
                case 2:
                F[0] = Q[3];
                F[1] = irho * Q[3] * Q[1];
                F[2] = irho * Q[3] * Q[2];
                F[3] = irho * Q[3] * Q[3] + p;
                F[4] = irho * Q[3] * (Q[4] + p);
            }
            """,
                stateless=stateless,
            )
        )
        solver.add_pde_term(
            exahype2.api.MaxEigenvalue(
                implementation="""
            // Eigenvalues of the Euler flux Jacobian in direction "normal" are
            // u_n +/- c, where u_n is the *velocity* component (momentum/density),
            // not the raw momentum component - several earlier ExaHyPE
            // applications/benchmarks used the raw momentum here by mistake, which
            // silently gives a wrong (and usually too small) CFL time step bound.
            const double irho  = 1.0 / Q[0];
            constexpr double gamma = 1.4;
            const double p = (gamma - 1.0) * (Q[4] - 0.5 * irho * (Q[1] * Q[1] + Q[2] * Q[2] + Q[3] * Q[3]));
            const double c = gamma * p * irho;

            switch (normal) {
                case 0:
                lambdaMax[0] = std::fmax(std::fabs(Q[1] * irho + c), std::fabs(Q[1] * irho - c));
                break;
                case 1:
                lambdaMax[0] = std::fmax(std::fabs(Q[2] * irho + c), std::fabs(Q[2] * irho - c));
                break;
                case 2:
                lambdaMax[0] = std::fmax(std::fabs(Q[3] * irho + c), std::fabs(Q[3] * irho - c));
                break;
            }
            """,
                stateless=stateless,
            )
        )
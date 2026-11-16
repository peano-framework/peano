#include "tarch/multicore/multicore.h"

#include <algorithm>
#include <limits>
#include <stdio.h>
#include <string.h>
#include <iostream>

#include "exagrype/ccz4/Kernels.h"
#include "tarch/Assertions.h"


void exagrype::ccz4::addKreissOligerDissipation(
  const double* __restrict__          QWithHalo,
  double* __restrict__                rightHandSide,
  const tarch::la::Vector<3, double>& patchSize,
  int                                 numberOfGridCellsPerPatchPerAxis,
  int                                 haloSize,
  int                                 unknowns,
  int                                 auxiliaryVariables,
  double                              sigma,
  const int*                          quantities,
  int                                 numberOfQuantities
) {
  assertion1(haloSize >= 3, haloSize);

  const int    n                     = numberOfGridCellsPerPatchPerAxis;
  const int    nWithHalo             = n + 2 * haloSize;
  const int    entriesPerGridCell    = unknowns + auxiliaryVariables;
  const double weights[7]            = {1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0};
  const int    strides[3]            = {entriesPerGridCell, nWithHalo * entriesPerGridCell, nWithHalo * nWithHalo * entriesPerGridCell};

  for (int z = 0; z < n; z++)
  for (int y = 0; y < n; y++)
  for (int x = 0; x < n; x++) {
    const int input  = (((z + haloSize) * nWithHalo + (y + haloSize)) * nWithHalo + (x + haloSize)) * entriesPerGridCell;
    const int output = ((z * n + y) * n + x) * unknowns;
    for (int i = 0; i < numberOfQuantities; i++) {
      const int quantity = quantities[i];
      for (int d = 0; d < 3; d++) {
        double dissipation = 0.0;
        for (int shift = -3; shift <= 3; shift++) {
          dissipation += weights[shift + 3] * QWithHalo[input + shift * strides[d] + quantity];
        }
        rightHandSide[output + quantity] += sigma / 64.0 * n / patchSize(d) * dissipation;
      }
    }
  }
}

namespace {
  /**
   * Shared part of enforceCCZ4constraints() and
   * enforceCCZ4constraintsOfEvolvedVariables(): rescale the conformal metric
   * Q[0..5] to unit determinant, remove the trace of Aex = Q[6..11], and
   * apply the floors of the lapse and phi. The inverse of the rescaled
   * metric is returned, so the caller does not have to compute it again.
   *
   * Symmetric matrices are computed on their upper triangle, and the six
   * entries of Q hold this triangle (0,1,2 / 3,4 / 5).
   */
  void enforceCCZ4constraintsOfMetricAndAex(double* Q, double g_contr[3][3]) {
    const double det   = Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] - Q[2]*Q[2]*Q[3];
    const double phisq = 1./std::cbrt(det);
    for (int i = 0; i < 6; i++) Q[i] *= phisq;

    const double invdet = 1./(Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] - Q[2]*Q[2]*Q[3]);
    const double g_contr_12 = -( Q[1]*Q[5]-Q[2]*Q[4])*invdet;
    const double g_contr_13 = -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet;
    const double g_contr_23 = -( Q[0]*Q[4]-Q[2]*Q[1])*invdet;
    g_contr[0][0] = (Q[3]*Q[5]-Q[4]*Q[4])*invdet;
    g_contr[1][1] = (Q[0]*Q[5]-Q[2]*Q[2])*invdet;
    g_contr[2][2] = (Q[0]*Q[3]-Q[1]*Q[1])*invdet;
    g_contr[0][1] = g_contr[1][0] = g_contr_12;
    g_contr[0][2] = g_contr[2][0] = g_contr_13;
    g_contr[1][2] = g_contr[2][1] = g_contr_23;

    // The upper triangle of the rescaled metric is Q[0..5], i.e. Q[i] below
    double* Aex = Q + 6;
    const double trAex = g_contr[0][0] * Aex[0]
                       + g_contr[1][1] * Aex[3]
                       + g_contr[2][2] * Aex[5]
                       + 2.0 * ( g_contr[0][1] * Aex[1]
                               + g_contr[0][2] * Aex[2]
                               + g_contr[1][2] * Aex[4] );
    for (int i = 0; i < 6; i++) Aex[i] -= 1./3. * trAex * Q[i];

    // Floors of the lapse and phi
    Q[16] = std::fmax(1e-16, Q[16]);
    Q[24] = std::fmax(1e-16, Q[24]);
  }
} // namespace


void exagrype::ccz4::enforceCCZ4constraintsOfEvolvedVariables(double * Q)
{
    double g_contr[3][3];
    enforceCCZ4constraintsOfMetricAndAex(Q, g_contr);
}


void exagrype::ccz4::enforceCCZ4constraints(double * Q)
{
    double g_contr[3][3];
    enforceCCZ4constraintsOfMetricAndAex(Q, g_contr);

    //
    // As suggested by our PRD referee, we also enforce the algebraic constraint that results from the first spatial derivative of the constraint
    // det \tilde{\gamma}_ij = 0, which leads to
    //
    // \tilde{\gamma}^{ij} D_kij = 0
    //
    // and is thus a condition of trace-freeness on all submatrices D_kij for k=1,2,3.
    // Q[37+6l .. 42+6l] hold the upper triangle of D_lij.
    //
    for (int l = 0; l < 3; l++)
    {
        double* DD = Q + 37 + 6 * l;
        const double trDD = g_contr[0][0] * DD[0]
                          + g_contr[1][1] * DD[3]
                          + g_contr[2][2] * DD[5]
                          + 2.0 * ( g_contr[0][1] * DD[1]
                                  + g_contr[0][2] * DD[2]
                                  + g_contr[1][2] * DD[4] );
        for (int i = 0; i < 6; i++) DD[i] -= 1./3. * Q[i] * trDD;
    }
}


void exagrype::ccz4::admconstraints(double* constraints, const double* const Q, const double* const gradQSerialised)
{
    constexpr int nVar(58);
    double gradQin[58][3] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // De-serialise input data and fill static array
    for (int normal=0; normal<3; normal++)
    for (int i=0; i<nVar; i++) gradQin[i][normal] = gradQSerialised[i+normal*nVar];

    // Note g_cov is symmetric
    const double g_cov[3][3] = { {Q[0], Q[1], Q[2]}, {Q[1], Q[3], Q[4]}, {Q[2], Q[4], Q[5]} };
    const double invdet = 1./( Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] -Q[2]*Q[2]*Q[3]);

    const double g_contr[3][3] = {
        { ( Q[3]*Q[5]-Q[4]*Q[4])*invdet, -( Q[1]*Q[5]-Q[2]*Q[4])*invdet, -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet},
        {-( Q[1]*Q[5]-Q[4]*Q[2])*invdet,  ( Q[0]*Q[5]-Q[2]*Q[2])*invdet, -( Q[0]*Q[4]-Q[2]*Q[1])*invdet},
        {-(-Q[1]*Q[4]+Q[3]*Q[2])*invdet, -( Q[0]*Q[4]-Q[1]*Q[2])*invdet,  ( Q[0]*Q[3]-Q[1]*Q[1])*invdet}
    };

    // NOTE Aex is symmetric
    double Aex[3][3] = { {Q[6], Q[7], Q[8]}, {Q[7], Q[9], Q[10]}, {Q[8], Q[10], Q[11]} };

    double traceA = 0;
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) traceA+=g_contr[i][j]*Aex[i][j];
    traceA *= 1./3;

    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Aex[i][j] -= traceA * g_cov[i][j];

    const double dAex[3][3][3] = {
        {{gradQin[6][0],gradQin[7][0],gradQin[8][0]}, {gradQin[7][0], gradQin[9][0], gradQin[10][0]},  {gradQin[8][0], gradQin[10][0], gradQin[11][0]}},
        {{gradQin[6][1],gradQin[7][1],gradQin[8][1]}, {gradQin[7][1], gradQin[9][1], gradQin[10][1]},  {gradQin[8][1], gradQin[10][1], gradQin[11][1]}},
        {{gradQin[6][2],gradQin[7][2],gradQin[8][2]}, {gradQin[7][2], gradQin[9][2], gradQin[10][2]},  {gradQin[8][2], gradQin[10][2], gradQin[11][2]}}
    };

    const double traceK = Q[23];
    const double dtraceK[3] = {gradQin[23][0], gradQin[23][1], gradQin[23][2]};

    const double phi = std::fmax(1e-2,Q[24]); 
    const double phi2 = phi*phi;
    const double PP[3] = {Q[55], Q[56], Q[57]};

    const double dPP[3][3] = {
        {gradQin[55][0],gradQin[56][0],gradQin[57][0]},
        {gradQin[55][1],gradQin[56][1],gradQin[57][1]},
        {gradQin[55][2],gradQin[56][2],gradQin[57][2]}
    };

    const double DD[3][3][3] = {
        {{Q[37], Q[38], Q[39]}, {Q[38], Q[40], Q[41]}, {Q[39], Q[41], Q[42]}},
        {{Q[43], Q[44], Q[45]}, {Q[44], Q[46], Q[47]}, {Q[45], Q[47], Q[48]}},
        {{Q[49], Q[50], Q[51]}, {Q[50], Q[52], Q[53]}, {Q[51], Q[53], Q[54]}}
    };
    const double dDD[3][3][3][3] = {
        {
                {
                        {gradQin[37][0],gradQin[38][0],gradQin[39][0]}, {gradQin[38][0],gradQin[40][0],gradQin[41][0]}, {gradQin[39][0],gradQin[41][0],gradQin[42][0]},
                },
                {
                        {gradQin[43][0],gradQin[44][0],gradQin[45][0]}, {gradQin[44][0],gradQin[46][0],gradQin[47][0]}, {gradQin[45][0],gradQin[47][0],gradQin[48][0]},
                },
                {
                        {gradQin[49][0],gradQin[50][0],gradQin[51][0]}, {gradQin[50][0],gradQin[52][0],gradQin[53][0]}, {gradQin[51][0],gradQin[53][0],gradQin[54][0]}
                }
        },
        {
                {
                        {gradQin[37][1],gradQin[38][1],gradQin[39][1]}, {gradQin[38][1],gradQin[40][1],gradQin[41][1]}, {gradQin[39][1],gradQin[41][1],gradQin[42][1]},
                },
                {
                        {gradQin[43][1],gradQin[44][1],gradQin[45][1]}, {gradQin[44][1],gradQin[46][1],gradQin[47][1]}, {gradQin[45][1],gradQin[47][1],gradQin[48][1]},
                },
                {
                        {gradQin[49][1],gradQin[50][1],gradQin[51][1]}, {gradQin[50][1],gradQin[52][1],gradQin[53][1]}, {gradQin[51][1],gradQin[53][1],gradQin[54][1]}
                }
        },
        {
                {
                        {gradQin[37][2],gradQin[38][2],gradQin[39][2]}, {gradQin[38][2],gradQin[40][2],gradQin[41][2]}, {gradQin[39][2],gradQin[41][2],gradQin[42][2]},
                },
                {
                        {gradQin[43][2],gradQin[44][2],gradQin[45][2]}, {gradQin[44][2],gradQin[46][2],gradQin[47][2]}, {gradQin[45][2],gradQin[47][2],gradQin[48][2]},
                },
                {
                        {gradQin[49][2],gradQin[50][2],gradQin[51][2]}, {gradQin[50][2],gradQin[52][2],gradQin[53][2]}, {gradQin[51][2],gradQin[53][2],gradQin[54][2]}
                }
        }
    };

    double dgup[3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int k = 0; k < 3; k++)
    for (int m = 0; m < 3; m++)
    for (int l = 0; l < 3; l++)
    for (int n = 0; n < 3; n++)
    for (int j = 0; j < 3; j++) dgup[k][m][l] -= g_contr[m][n]*g_contr[j][l]*2*DD[k][n][j];

    double Kex[3][3]={ 0 };
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Kex[i][j]=Aex[i][j]/phi2 + (1./3)*traceK*g_cov[i][j]/phi2;
    double Kmix[3][3]={0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int u = 0; u < 3; u++) Kmix[i][j] += phi2*g_contr[i][u] * Kex[u][j];
    double Kup[3][3]={0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int u = 0; u < 3; u++) Kup[i][j] += phi2*g_contr[i][u] * Kmix[j][u]; // Note the transposition is in the indices

    double Christoffel[3][3][3]       = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double Christoffel_tilde[3][3][3]       = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++) {
        Christoffel_tilde[i][j][k] += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] );
        Christoffel[i][j][k]  += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] ) - g_contr[k][l]/phi * ( g_cov[j][l] * PP[i] + g_cov[i][l] * PP[j] - g_cov[i][j] * PP[l] );
    }

    double dChristoffel[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double dChristoffel_tilde[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
 
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++)
    {
        dChristoffel[k][i][j][m] += 0.5*g_contr[m][l] * ( //npc part
            dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]
            - g_cov[j][l]*(dPP[k][i] + dPP[i][k])/phi - g_cov[i][l]*(dPP[k][j]+dPP[j][k])/phi +  g_cov[i][j]*(dPP[k][l]+dPP[l][k])/phi )
            +dgup[k][m][l]*(DD[i][j][l]+DD[j][i][l]-DD[l][i][j])  //src part
            -dgup[k][m][l]*(g_cov[j][l]*PP[i]+g_cov[i][l]*PP[j]-g_cov[i][j]*PP[l])/phi
            -2*g_contr[m][l]*(DD[k][j][l]*PP[i]+DD[k][i][l]*PP[j]-DD[k][i][j]*PP[l])/phi
            +g_contr[m][l]*(g_cov[j][l]*PP[i]*PP[k]+g_cov[i][l]*PP[j]*PP[k]-g_cov[i][j]*PP[k]*PP[l])/phi2;

        dChristoffel_tilde[k][i][j][m] += dgup[k][m][l]*(DD[i][j][l]+DD[j][i][l]-DD[l][i][j])
            +0.5*g_contr[m][l]*(dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]);
    }

    double Riemann[3][3][3][3] = {0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++){
        Riemann[i][k][j][m] = dChristoffel[k][i][j][m] - dChristoffel[j][i][k][m];
        for (int l = 0; l < 3; l++){
            Riemann[i][k][j][m] += Christoffel[i][j][l]*Christoffel[l][k][m]-Christoffel[i][k][l]*Christoffel[l][j][m];
        }
    }

    double Ricci[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++)
    for (int l = 0; l < 3; l++) Ricci[m][n] += Riemann[m][l][n][l];

    double R=0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) R += phi2*g_contr[i][j]*Ricci[i][j];
    double Kupdown=0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) Kupdown += Kex[i][j]*Kup[i][j];

// following is added to calculate nabla_i Z^i
    double Gtilde[3] = {0,0,0};
    for (int l = 0; l < 3; l++)
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) Gtilde[i] += g_contr[j][l] * Christoffel_tilde[j][l][i];

    const double Ghat[3] = {Q[13], Q[14], Q[15]};

    const double dGhat[3][3] = {
        {gradQin[13][0],gradQin[14][0],gradQin[15][0]},
        {gradQin[13][1],gradQin[14][1],gradQin[15][1]},
        {gradQin[13][2],gradQin[14][2],gradQin[15][2]}
    };

    double dGtilde[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int j = 0; j < 3; j++)
    for (int l = 0; l < 3; l++) dGtilde[k][i] += dgup[k][j][l]*Christoffel_tilde[j][l][i]+g_contr[j][l]*dChristoffel_tilde[k][j][l][i];

    double Z[3] = {0,0,0}; // Matrix vector multiplications
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Z[i] += 0.5* g_cov[i][j]* (Ghat[j] - Gtilde[j]);//needed here

    double dZ[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++) dZ[k][i] += DD[k][i][j]*(Ghat[j]-Gtilde[j])+0.5*g_cov[i][j]*(dGhat[k][j]-dGtilde[k][j]);

    double nablaZ[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    {
      nablaZ[i][j] = dZ[i][j];
      for (int k = 0; k < 3; k++) nablaZ[i][j] -= Christoffel[i][j][k]*Z[k];
    }

    double nablaZcontracted=0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) nablaZcontracted += g_contr[i][j]*nablaZ[i][j]; // TODO fuse these steps
    nablaZcontracted*=phi2;

//end here

    double Ham=0;
    Ham = R-Kupdown+traceK*traceK;

    double dKex[3][3][3]={0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) dKex[k][i][j]  = (1.0/phi2)*(dAex[k][i][j]+(1./3)*g_cov[i][j]*dtraceK[k]+(2./3)*traceK*DD[k][i][j])-2*Kex[i][j]*PP[k]/phi;

    double Mom[3]={0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int l = 0; l < 3; l++){
        Mom[i] += phi2*g_contr[j][l]*(dKex[l][i][j]-dKex[i][j][l]);
        for (int m = 0; m < 3; m++){
            Mom[i] += phi2*g_contr[j][l]*(-Christoffel[j][l][m]*Kex[m][i]+Christoffel[j][i][m]*Kex[m][l]);
        }
    }

    tarch::memset(constraints, 0, 4*sizeof(double));

    constraints[0]  =   Ham;        //58
    constraints[1]  =   Mom[0];     //59
    constraints[2]  =   Mom[1];     //60
    constraints[3]  =   Mom[2];     //61
    //constraints[4]  =   R;          //62
    //constraints[5]  =   Kupdown;    //63
    //constraints[6]  =   nablaZcontracted;   //64
    //constraints[4]  =   1.0/invdet-1.0;
    //constraints[5]  =   traceA;
}


void exagrype::ccz4::Psi4Calc(double* Psi4, const double* const Q, const double* const gradQSerialised, double* coor)
{
    constexpr int nVar(58);
    double gradQin[58][3] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // De-serialise input data and fill static array
    for (int normal=0; normal<3; normal++)
    for (int i=0; i<nVar; i++) gradQin[i][normal] = gradQSerialised[i+normal*nVar];

    // Note g_cov is symmetric
    const double g_cov[3][3] = { {Q[0], Q[1], Q[2]}, {Q[1], Q[3], Q[4]}, {Q[2], Q[4], Q[5]} };
    const double invdet = 1./( Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] -Q[2]*Q[2]*Q[3]);

    const double g_contr[3][3] = {
        { ( Q[3]*Q[5]-Q[4]*Q[4])*invdet, -( Q[1]*Q[5]-Q[2]*Q[4])*invdet, -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet},
        {-( Q[1]*Q[5]-Q[4]*Q[2])*invdet,  ( Q[0]*Q[5]-Q[2]*Q[2])*invdet, -( Q[0]*Q[4]-Q[2]*Q[1])*invdet},
        {-(-Q[1]*Q[4]+Q[3]*Q[2])*invdet, -( Q[0]*Q[4]-Q[1]*Q[2])*invdet,  ( Q[0]*Q[3]-Q[1]*Q[1])*invdet}
    };

    // NOTE Aex is symmetric
    double Aex[3][3] = { {Q[6], Q[7], Q[8]}, {Q[7], Q[9], Q[10]}, {Q[8], Q[10], Q[11]} };

    double traceA = 0;
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) traceA+=g_contr[i][j]*Aex[i][j];
    traceA *= 1./3;

    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Aex[i][j] -= traceA * g_cov[i][j];

    const double dAex[3][3][3] = {
        {{gradQin[6][0],gradQin[7][0],gradQin[8][0]}, {gradQin[7][0], gradQin[9][0], gradQin[10][0]},  {gradQin[8][0], gradQin[10][0], gradQin[11][0]}},
        {{gradQin[6][1],gradQin[7][1],gradQin[8][1]}, {gradQin[7][1], gradQin[9][1], gradQin[10][1]},  {gradQin[8][1], gradQin[10][1], gradQin[11][1]}},
        {{gradQin[6][2],gradQin[7][2],gradQin[8][2]}, {gradQin[7][2], gradQin[9][2], gradQin[10][2]},  {gradQin[8][2], gradQin[10][2], gradQin[11][2]}}
    };

    const double traceK = Q[23];
    const double dtraceK[3] = {gradQin[23][0], gradQin[23][1], gradQin[23][2]};

    const double phi = std::fmax(1e-2,Q[24]); ;
    const double phi2 = phi*phi;
    const double PP[3] = {Q[55], Q[56], Q[57]};

    const double dPP[3][3] = {
        {gradQin[55][0],gradQin[56][0],gradQin[57][0]},
        {gradQin[55][1],gradQin[56][1],gradQin[57][1]},
        {gradQin[55][2],gradQin[56][2],gradQin[57][2]}
    };

    const double DD[3][3][3] = {
        {{Q[37], Q[38], Q[39]}, {Q[38], Q[40], Q[41]}, {Q[39], Q[41], Q[42]}},
        {{Q[43], Q[44], Q[45]}, {Q[44], Q[46], Q[47]}, {Q[45], Q[47], Q[48]}},
        {{Q[49], Q[50], Q[51]}, {Q[50], Q[52], Q[53]}, {Q[51], Q[53], Q[54]}}
    };
    const double dDD[3][3][3][3] = {
        {
                {
                        {gradQin[37][0],gradQin[38][0],gradQin[39][0]}, {gradQin[38][0],gradQin[40][0],gradQin[41][0]}, {gradQin[39][0],gradQin[41][0],gradQin[42][0]},
                },
                {
                        {gradQin[43][0],gradQin[44][0],gradQin[45][0]}, {gradQin[44][0],gradQin[46][0],gradQin[47][0]}, {gradQin[45][0],gradQin[47][0],gradQin[48][0]},
                },
                {
                        {gradQin[49][0],gradQin[50][0],gradQin[51][0]}, {gradQin[50][0],gradQin[52][0],gradQin[53][0]}, {gradQin[51][0],gradQin[53][0],gradQin[54][0]}
                }
        },
        {
                {
                        {gradQin[37][1],gradQin[38][1],gradQin[39][1]}, {gradQin[38][1],gradQin[40][1],gradQin[41][1]}, {gradQin[39][1],gradQin[41][1],gradQin[42][1]},
                },
                {
                        {gradQin[43][1],gradQin[44][1],gradQin[45][1]}, {gradQin[44][1],gradQin[46][1],gradQin[47][1]}, {gradQin[45][1],gradQin[47][1],gradQin[48][1]},
                },
                {
                        {gradQin[49][1],gradQin[50][1],gradQin[51][1]}, {gradQin[50][1],gradQin[52][1],gradQin[53][1]}, {gradQin[51][1],gradQin[53][1],gradQin[54][1]}
                }
        },
        {
                {
                        {gradQin[37][2],gradQin[38][2],gradQin[39][2]}, {gradQin[38][2],gradQin[40][2],gradQin[41][2]}, {gradQin[39][2],gradQin[41][2],gradQin[42][2]},
                },
                {
                        {gradQin[43][2],gradQin[44][2],gradQin[45][2]}, {gradQin[44][2],gradQin[46][2],gradQin[47][2]}, {gradQin[45][2],gradQin[47][2],gradQin[48][2]},
                },
                {
                        {gradQin[49][2],gradQin[50][2],gradQin[51][2]}, {gradQin[50][2],gradQin[52][2],gradQin[53][2]}, {gradQin[51][2],gradQin[53][2],gradQin[54][2]}
                }
        }
    };

    double dgup[3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int k = 0; k < 3; k++)
    for (int m = 0; m < 3; m++)
    for (int l = 0; l < 3; l++)
    for (int n = 0; n < 3; n++)
    for (int j = 0; j < 3; j++) dgup[k][m][l] -= g_contr[m][n]*g_contr[j][l]*2*DD[k][n][j];

    double Kex[3][3]={ 0 };
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Kex[i][j]=Aex[i][j]/phi2 + (1./3)*traceK*g_cov[i][j]/phi2;

    double Christoffel[3][3][3]       = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double Christoffel_tilde[3][3][3]       = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++) {
        Christoffel_tilde[i][j][k] += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] );
        Christoffel[i][j][k]  += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] ) - g_contr[k][l]/phi * ( g_cov[j][l] * PP[i] + g_cov[i][l] * PP[j] - g_cov[i][j] * PP[l] );
    }

    double dChristoffel[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double dChristoffel_tilde[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
 
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++)
    {
        dChristoffel[k][i][j][m] += 0.5*g_contr[m][l] * ( //npc part
            dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]
            - g_cov[j][l]*(dPP[k][i] + dPP[i][k])/phi - g_cov[i][l]*(dPP[k][j]+dPP[j][k])/phi +  g_cov[i][j]*(dPP[k][l]+dPP[l][k])/phi )
            +dgup[k][m][l]*(DD[i][j][l]+DD[j][i][l]-DD[l][i][j])  //src part
            -dgup[k][m][l]*(g_cov[j][l]*PP[i]+g_cov[i][l]*PP[j]-g_cov[i][j]*PP[l])/phi
            -2*g_contr[m][l]*(DD[k][j][l]*PP[i]+DD[k][i][l]*PP[j]-DD[k][i][j]*PP[l])/phi
            +g_contr[m][l]*(g_cov[j][l]*PP[i]*PP[k]+g_cov[i][l]*PP[j]*PP[k]-g_cov[i][j]*PP[k]*PP[l])/phi2;

        dChristoffel_tilde[k][i][j][m] += dgup[k][m][l]*(DD[i][j][l]+DD[j][i][l]-DD[l][i][j])
            +0.5*g_contr[m][l]*(dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]);
    }

    double Gtilde[3] = {0,0,0};
    for (int l = 0; l < 3; l++)
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) Gtilde[i] += g_contr[j][l] * Christoffel_tilde[j][l][i];

    const double Theta = Q[12]; // needed later
    const double Ghat[3] = {Q[13], Q[14], Q[15]};

    double Z[3] = {0,0,0}; // Matrix vector multiplications
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Z[i] += 0.5* g_cov[i][j]* (Ghat[j] - Gtilde[j]);

    const double dGhat[3][3] = {
        {gradQin[13][0],gradQin[14][0],gradQin[15][0]},
        {gradQin[13][1],gradQin[14][1],gradQin[15][1]},
        {gradQin[13][2],gradQin[14][2],gradQin[15][2]}
    };

    double dGtilde[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int j = 0; j < 3; j++)
    for (int l = 0; l < 3; l++) dGtilde[k][i] += dgup[k][j][l]*Christoffel_tilde[j][l][i]+g_contr[j][l]*dChristoffel_tilde[k][j][l][i];

    double dZ[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++) dZ[k][i] += DD[k][i][j]*(Ghat[j]-Gtilde[j])+0.5*g_cov[i][j]*(dGhat[k][j]-dGtilde[k][j]);

    double nablaZ[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    {
      nablaZ[i][j] = dZ[i][j];
      for (int k = 0; k < 3; k++) nablaZ[i][j] -= Christoffel[i][j][k]*Z[k];
    }


    double Riemann[3][3][3][3] = {0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++){
        Riemann[i][k][j][m] = dChristoffel[k][i][j][m] - dChristoffel[j][i][k][m];
        for (int l = 0; l < 3; l++){
            Riemann[i][k][j][m] += Christoffel[i][j][l]*Christoffel[l][k][m]-Christoffel[i][k][l]*Christoffel[l][j][m];
        }
    }

    double Ricci[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++)
    for (int l = 0; l < 3; l++) Ricci[m][n] += Riemann[m][l][n][l];
    
    double dKex[3][3][3]   = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};  //derivative  \partial_k K_ij
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++) dKex[k][i][j]  = (1.0/phi2)*(dAex[k][i][j]+(1./3)*g_cov[i][j]*dtraceK[k]+(2./3)*traceK*DD[k][i][j])-2*Kex[i][j]*PP[k]/phi;
    
    double Cov_dKex[3][3][3]   = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};  //covariant derivative \nabla_i K_jk 
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int m = 0; m < 3; m++) Cov_dKex[i][j][k] += dKex[i][j][k]-Christoffel[i][k][m]*Kex[j][m]-Christoffel[i][j][m]*Kex[m][k];
    
	const double detgd=(1.0/(phi2*phi2*phi2))*( Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] -Q[2]*Q[2]*Q[3]); //determinant of \gamma_ij
	
	double eps_lc_u[3][3][3]   = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; //Levi-Civita tensor
    double eps_lc_symbol[3][3][3]   = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; //Levi-Civita symbol
	eps_lc_u[0][1][2] = 1/pow(detgd,0.5); eps_lc_u[1][2][0] = 1/pow(detgd,0.5); eps_lc_u[2][0][1] = 1/pow(detgd,0.5);
	eps_lc_u[2][1][0] =-1/pow(detgd,0.5); eps_lc_u[0][2][1] =-1/pow(detgd,0.5); eps_lc_u[1][0][2] =-1/pow(detgd,0.5);
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++) {
        eps_lc_symbol[i][j][k] = eps_lc_u[i][j][k] * pow(detgd,0.5);
        if (detgd<0){ eps_lc_u[i][j][k]= - eps_lc_u[i][j][k]; }
    }

	
	//calculate the orthonormal basis
    //mainly follow the paper: https://arxiv.org/pdf/gr-qc/0610128v1.pdf
	if ((coor[0]*coor[0]+coor[1]*coor[1])<1e-10) {coor[0]+=1e-10;}
	double u_vec[3]={coor[0], coor[1], coor[2]};//r-direction
	double v_vec[3]={-coor[2], coor[1], 0.0}; //phi-direction
	//double w_vec[3]={coor[0]*coor[2],coor[1]*coor[2],-coor[0]*coor[0]-coor[1]*coor[1]};//theta-direction //??
    double w_vec[3]={0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++) w_vec[i] += phi2*g_contr[i][j]*eps_lc_symbol[j][k][l]*v_vec[k]*u_vec[l];

	
	double dotp1=0;
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) dotp1 += g_cov[i][j]*v_vec[i]*v_vec[j]/phi2;
	for (int a = 0; a < 3; a++) v_vec[a] = v_vec[a] / pow(dotp1,0.5);
	
	dotp1=0;
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) dotp1 += g_cov[i][j]*v_vec[i]*u_vec[j]/phi2;
	for (int a = 0; a < 3; a++) u_vec[a] = u_vec[a] - dotp1* v_vec[a];	
	
	dotp1=0;
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) dotp1 += g_cov[i][j]*u_vec[i]*u_vec[j]/phi2;
	for (int a = 0; a < 3; a++) u_vec[a] = u_vec[a] / pow(dotp1,0.5);
	
	dotp1=0;
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) dotp1 += g_cov[i][j]*v_vec[i]*w_vec[j]/phi2;	
	double dotp2=0;
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) dotp2 += g_cov[i][j]*u_vec[i]*w_vec[j]/phi2;
    
	for (int a = 0; a < 3; a++) w_vec[a] = u_vec[a] - dotp1 * v_vec[a] - dotp2 * u_vec[a];		
		
	dotp1=0;
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) dotp1 += g_cov[i][j]*w_vec[i]*w_vec[j]/phi2;
	for (int a = 0; a < 3; a++) w_vec[a] = w_vec[a] / pow(dotp1,0.5);
	
	//EB part of Weyl
    //reference: https://academic.oup.com/book/9640/chapter/156750056
    //8.3.15 and 8.3.16
	double elec[3][3] = {0,0,0,0,0,0,0,0,0};
	double mag[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++){
    	elec[i][j]=Ricci[i][j];
		for (int m = 0; m < 3; m++)
		for (int k = 0; k < 3; k++){
			elec[i][j] += phi2*g_contr[m][k]*(Kex[i][j]*Kex[m][k]-Kex[i][m]*Kex[k][j])-Kex[i][j]*Theta + 0.5*(nablaZ[i][j]+nablaZ[j][i]);
			for (int l = 0; l < 3; l++){
				mag[i][j] += g_cov[i][l] * eps_lc_u[l][k][m] * Cov_dKex[k][m][j] + g_cov[j][l] * eps_lc_u[l][k][m] * Cov_dKex[k][m][i];
			}
		}
		mag[i][j]=mag[i][j]/(2*phi2);
	}
	
	double electr=0, magtr=0;
	for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++){
    	electr += phi2*g_contr[i][j]*elec[i][j];
		magtr  += phi2*g_contr[i][j]*mag[i][j];
	}
	for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++){
		elec[i][j] = elec[i][j] - g_cov[i][j]*electr / (3*phi2);
		mag[i][j]  = mag[i][j]  - g_cov[i][j]*magtr  / (3*phi2);
	}

    //use 8.6.17 and 8.6.18
    //notice we use \bar{m}^i={1 \over \sqrt{2} }(w^i- i v^i)
    Psi4[0]=0.0; Psi4[1]=0.0;
	for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++){
    	//Real part for Psi4
		Psi4[0] += 0.5*elec[i][j]*(w_vec[i]*w_vec[j]-v_vec[i]*v_vec[j])-0.5*mag[i][j]*(v_vec[i]*w_vec[j]+v_vec[j]*w_vec[i]);
		//imaginary part for Psi4
		Psi4[1] += - 0.5*elec[i][j]*(v_vec[i]*w_vec[j]+v_vec[j]*w_vec[i])-0.5*mag[i][j]*(w_vec[i]*w_vec[j]-v_vec[i]*v_vec[j]);
	}
	
}


void exagrype::ccz4::TestingOutput(double* terms, const double* const Q, const double* const gradQSerialised)
{
    constexpr int nVar(58);
    double gradQin[58][3] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // De-serialise input data and fill static array
    for (int normal=0; normal<3; normal++)
    for (int i=0; i<nVar; i++) gradQin[i][normal] = gradQSerialised[i+normal*nVar];

    // Note g_cov is symmetric
    const double g_cov[3][3] = { {Q[0], Q[1], Q[2]}, {Q[1], Q[3], Q[4]}, {Q[2], Q[4], Q[5]} };
    const double invdet = 1./( Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] -Q[2]*Q[2]*Q[3]);

    const double g_contr[3][3] = {
        { ( Q[3]*Q[5]-Q[4]*Q[4])*invdet, -( Q[1]*Q[5]-Q[2]*Q[4])*invdet, -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet},
        {-( Q[1]*Q[5]-Q[4]*Q[2])*invdet,  ( Q[0]*Q[5]-Q[2]*Q[2])*invdet, -( Q[0]*Q[4]-Q[2]*Q[1])*invdet},
        {-(-Q[1]*Q[4]+Q[3]*Q[2])*invdet, -( Q[0]*Q[4]-Q[1]*Q[2])*invdet,  ( Q[0]*Q[3]-Q[1]*Q[1])*invdet}
    };

    // NOTE Aex is symmetric
    double Aex[3][3] = { {Q[6], Q[7], Q[8]}, {Q[7], Q[9], Q[10]}, {Q[8], Q[10], Q[11]} };

    double traceA = 0;
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) traceA+=g_contr[i][j]*Aex[i][j];
    traceA *= 1./3;

    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Aex[i][j] -= traceA * g_cov[i][j];

    const double dAex[3][3][3] = {
        {{gradQin[6][0],gradQin[7][0],gradQin[8][0]}, {gradQin[7][0], gradQin[9][0], gradQin[10][0]},  {gradQin[8][0], gradQin[10][0], gradQin[11][0]}},
        {{gradQin[6][1],gradQin[7][1],gradQin[8][1]}, {gradQin[7][1], gradQin[9][1], gradQin[10][1]},  {gradQin[8][1], gradQin[10][1], gradQin[11][1]}},
        {{gradQin[6][2],gradQin[7][2],gradQin[8][2]}, {gradQin[7][2], gradQin[9][2], gradQin[10][2]},  {gradQin[8][2], gradQin[10][2], gradQin[11][2]}}
    };

    const double traceK = Q[23];
    const double dtraceK[3] = {gradQin[23][0], gradQin[23][1], gradQin[23][2]};

    const double phi = std::fmax(1e-2,Q[24]); 
    const double phi2 = phi*phi;
    const double PP[3] = {Q[55], Q[56], Q[57]};

    const double dPP[3][3] = {
        {gradQin[55][0],gradQin[56][0],gradQin[57][0]},
        {gradQin[55][1],gradQin[56][1],gradQin[57][1]},
        {gradQin[55][2],gradQin[56][2],gradQin[57][2]}
    };

    const double DD[3][3][3] = {
        {{Q[37], Q[38], Q[39]}, {Q[38], Q[40], Q[41]}, {Q[39], Q[41], Q[42]}},
        {{Q[43], Q[44], Q[45]}, {Q[44], Q[46], Q[47]}, {Q[45], Q[47], Q[48]}},
        {{Q[49], Q[50], Q[51]}, {Q[50], Q[52], Q[53]}, {Q[51], Q[53], Q[54]}}
    };
    const double dDD[3][3][3][3] = {
        {
                {
                        {gradQin[37][0],gradQin[38][0],gradQin[39][0]}, {gradQin[38][0],gradQin[40][0],gradQin[41][0]}, {gradQin[39][0],gradQin[41][0],gradQin[42][0]},
                },
                {
                        {gradQin[43][0],gradQin[44][0],gradQin[45][0]}, {gradQin[44][0],gradQin[46][0],gradQin[47][0]}, {gradQin[45][0],gradQin[47][0],gradQin[48][0]},
                },
                {
                        {gradQin[49][0],gradQin[50][0],gradQin[51][0]}, {gradQin[50][0],gradQin[52][0],gradQin[53][0]}, {gradQin[51][0],gradQin[53][0],gradQin[54][0]}
                }
        },
        {
                {
                        {gradQin[37][1],gradQin[38][1],gradQin[39][1]}, {gradQin[38][1],gradQin[40][1],gradQin[41][1]}, {gradQin[39][1],gradQin[41][1],gradQin[42][1]},
                },
                {
                        {gradQin[43][1],gradQin[44][1],gradQin[45][1]}, {gradQin[44][1],gradQin[46][1],gradQin[47][1]}, {gradQin[45][1],gradQin[47][1],gradQin[48][1]},
                },
                {
                        {gradQin[49][1],gradQin[50][1],gradQin[51][1]}, {gradQin[50][1],gradQin[52][1],gradQin[53][1]}, {gradQin[51][1],gradQin[53][1],gradQin[54][1]}
                }
        },
        {
                {
                        {gradQin[37][2],gradQin[38][2],gradQin[39][2]}, {gradQin[38][2],gradQin[40][2],gradQin[41][2]}, {gradQin[39][2],gradQin[41][2],gradQin[42][2]},
                },
                {
                        {gradQin[43][2],gradQin[44][2],gradQin[45][2]}, {gradQin[44][2],gradQin[46][2],gradQin[47][2]}, {gradQin[45][2],gradQin[47][2],gradQin[48][2]},
                },
                {
                        {gradQin[49][2],gradQin[50][2],gradQin[51][2]}, {gradQin[50][2],gradQin[52][2],gradQin[53][2]}, {gradQin[51][2],gradQin[53][2],gradQin[54][2]}
                }
        }
    };

    double dgup[3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int k = 0; k < 3; k++)
    for (int m = 0; m < 3; m++)
    for (int l = 0; l < 3; l++)
    for (int n = 0; n < 3; n++)
    for (int j = 0; j < 3; j++) dgup[k][m][l] -= g_contr[m][n]*g_contr[j][l]*2*DD[k][n][j];

    double Kex[3][3]={ 0 };
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Kex[i][j]=Aex[i][j]/phi2 + (1./3)*traceK*g_cov[i][j]/phi2;
    double Kmix[3][3]={0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int u = 0; u < 3; u++) Kmix[i][j] += phi2*g_contr[i][u] * Kex[u][j];
    double Kup[3][3]={0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int u = 0; u < 3; u++) Kup[i][j] += phi2*g_contr[i][u] * Kmix[j][u]; // Note the transposition is in the indices

    double Christoffel[3][3][3]       = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double Christoffel_tilde[3][3][3]       = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++) {
        Christoffel_tilde[i][j][k] += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] );
        Christoffel[i][j][k]  += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] ) - g_contr[k][l]/phi * ( g_cov[j][l] * PP[i] + g_cov[i][l] * PP[j] - g_cov[i][j] * PP[l] );
    }

    double dChristoffel[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double dChristoffel_tilde[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
 
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++)
    {
        dChristoffel[k][i][j][m] += 0.5*g_contr[m][l] * ( //npc part
            dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]
            - g_cov[j][l]*(dPP[k][i] + dPP[i][k])/phi - g_cov[i][l]*(dPP[k][j]+dPP[j][k])/phi +  g_cov[i][j]*(dPP[k][l]+dPP[l][k])/phi )
            +dgup[k][m][l]*(DD[i][j][l]+DD[j][i][l]-DD[l][i][j])  //src part
            -dgup[k][m][l]*(g_cov[j][l]*PP[i]+g_cov[i][l]*PP[j]-g_cov[i][j]*PP[l])/phi
            -2*g_contr[m][l]*(DD[k][j][l]*PP[i]+DD[k][i][l]*PP[j]-DD[k][i][j]*PP[l])/phi
            +g_contr[m][l]*(g_cov[j][l]*PP[i]*PP[k]+g_cov[i][l]*PP[j]*PP[k]-g_cov[i][j]*PP[k]*PP[l])/phi2;

        dChristoffel_tilde[k][i][j][m] += dgup[k][m][l]*(DD[i][j][l]+DD[j][i][l]-DD[l][i][j])
            +0.5*g_contr[m][l]*(dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]);
    }

    double Riemann[3][3][3][3] = {0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++){
        Riemann[i][k][j][m] = dChristoffel[k][i][j][m] - dChristoffel[j][i][k][m];
        for (int l = 0; l < 3; l++){
            Riemann[i][k][j][m] += Christoffel[i][j][l]*Christoffel[l][k][m]-Christoffel[i][k][l]*Christoffel[l][j][m];
        }
    }

    double Ricci[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++)
    for (int l = 0; l < 3; l++) Ricci[m][n] += Riemann[m][l][n][l];

    double R=0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) R += phi2*g_contr[i][j]*Ricci[i][j];
    double Kupdown=0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) Kupdown += Kex[i][j]*Kup[i][j];

// following is added to calculate nabla_i Z^i
    double Gtilde[3] = {0,0,0};
    for (int l = 0; l < 3; l++)
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) Gtilde[i] += g_contr[j][l] * Christoffel_tilde[j][l][i];

    const double Ghat[3] = {Q[13], Q[14], Q[15]};

    const double dGhat[3][3] = {
        {gradQin[13][0],gradQin[14][0],gradQin[15][0]},
        {gradQin[13][1],gradQin[14][1],gradQin[15][1]},
        {gradQin[13][2],gradQin[14][2],gradQin[15][2]}
    };

    double dGtilde[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int j = 0; j < 3; j++)
    for (int l = 0; l < 3; l++) dGtilde[k][i] += dgup[k][j][l]*Christoffel_tilde[j][l][i]+g_contr[j][l]*dChristoffel_tilde[k][j][l][i];

    double Z[3] = {0,0,0}; // Matrix vector multiplications
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Z[i] += 0.5* g_cov[i][j]* (Ghat[j] - Gtilde[j]);//needed here

    double dZ[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++) dZ[k][i] += DD[k][i][j]*(Ghat[j]-Gtilde[j])+0.5*g_cov[i][j]*(dGhat[k][j]-dGtilde[k][j]);

    double nablaZ[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    {
      nablaZ[i][j] = dZ[i][j];
      for (int k = 0; k < 3; k++) nablaZ[i][j] -= Christoffel[i][j][k]*Z[k];
    }

    double nablaZcontracted=0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) nablaZcontracted += g_contr[i][j]*nablaZ[i][j]; // TODO fuse these steps
    nablaZcontracted*=phi2;

//end here

    double Ham=0;
    Ham = R-Kupdown+traceK*traceK;

    double dKex[3][3][3]={0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++) dKex[k][i][j]  = (1.0/phi2)*(dAex[k][i][j]+(1./3)*g_cov[i][j]*dtraceK[k]+(2./3)*traceK*DD[k][i][j])-2*Kex[i][j]*PP[k]/phi;

    double Mom[3]={0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int l = 0; l < 3; l++){
        Mom[i] += phi2*g_contr[j][l]*(dKex[l][i][j]-dKex[i][j][l]);
        for (int m = 0; m < 3; m++){
            Mom[i] += phi2*g_contr[j][l]*(-Christoffel[j][l][m]*Kex[m][i]+Christoffel[j][i][m]*Kex[m][l]);
        }
    }

    tarch::memset(terms, 0, 15*sizeof(double));
    terms[0]  =   Ham;        //58
    terms[1]  =   Mom[0];     //59
    terms[2]  =   Mom[1];     //60
    terms[3]  =   Mom[2];     //61
    terms[4]  =   R;          //62
    terms[5]  =   Kupdown;    //63
    terms[6]  =   nablaZcontracted;   //64
    //calculate the terms for Px
    const double alpha = std::fmax(1e-4,Q[16]);
    const double dBB[3][3][3] = {
        {
            { gradQin[28][0], gradQin[29][0], gradQin[30][0]}, { gradQin[31][0], gradQin[32][0], gradQin[33][0]}, { gradQin[34][0], gradQin[35][0], gradQin[36][0]}
        },
        {
            { gradQin[28][1], gradQin[29][1], gradQin[30][1]}, { gradQin[31][1], gradQin[32][1], gradQin[33][1]}, { gradQin[34][1], gradQin[35][1], gradQin[36][1]}
        },
        {
            { gradQin[28][2], gradQin[29][2], gradQin[30][2]}, { gradQin[31][2], gradQin[32][2], gradQin[33][2]}, { gradQin[34][2], gradQin[35][2], gradQin[36][2]}
        }
    };
    const double BB[3][3] = {
        {Q[28], Q[29], Q[30]}, {Q[31], Q[32], Q[33]}, {Q[34], Q[35], Q[36]}
    };
    double TrBB = BB[0][0] + BB[1][1] + BB[2][2];
    const double AA[3] = {Q[25], Q[26], Q[27]};

    for (int l = 0; l < 3; l++) terms[7]+=BB[0][l]*PP[l]; //65
    terms[8] = (1.0/3.0)*alpha*traceK*PP[0]; //66
    terms[9] = -(1.0/3.0)*TrBB*PP[0]; //67
    terms[10]= (1.0/3.0)*phi*alpha*dtraceK[0]; //68
    terms[11]= (1.0/3.0)*phi*traceK*AA[0]; //69
    for (int l = 0; l < 3; l++) terms[12] -= 1./6.*phi*(dBB[0][l][l] + dBB[l][0][l]);  //70
    double temp=0;
    for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++) temp += g_contr[m][n]*dAex[0][m][n];
    terms[13] += 1./3.*alpha*temp; //71
    temp=0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) temp += dgup[0][i][j]*Aex[i][j];
    terms[14] += 1./3.*alpha*temp;  //72

    //constraints[4]  =   1.0/invdet-1.0;
    //constraints[5]  =   traceA;
}
void exagrype::ccz4::ThetaOutputNCP(double* NCPterm, const double* const Q, const double* const gradQSerialised, int normal)
{
    const double alpha = std::exp(std::fmax(-20., std::fmin(20.,Q[16])));
    //printf("alpha %f\n",alpha);
    const double alpha2 = alpha*alpha;
    double ga  = 1.0;
    double gaa = 0.0;
    ga  =  2./alpha;
    gaa = -ga/alpha;
    const double Kex = Q[23];

    constexpr int nVar(58);

    double gradQin[58][3] ={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int i=0; i<nVar; i++) {
      gradQin[i][normal] = gradQSerialised[i];
      //if (gradQSerialised[i]>0.0) std::cout<< gradQSerialised[i] << std::endl; 
    }

    // Note g_cov is symmetric
    const double g_cov[3][3] = { {Q[0], Q[1], Q[2]}, {Q[1], Q[3], Q[4]}, {Q[2], Q[4], Q[5]} };
    const double invdet = 1./( Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] -Q[2]*Q[2]*Q[3]);

    const double g_contr[3][3] = {
        { ( Q[3]*Q[5]-Q[4]*Q[4])*invdet, -( Q[1]*Q[5]-Q[2]*Q[4])*invdet, -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet},
        {-( Q[1]*Q[5]-Q[4]*Q[2])*invdet,  ( Q[0]*Q[5]-Q[2]*Q[2])*invdet, -( Q[0]*Q[4]-Q[2]*Q[1])*invdet},
        {-(-Q[1]*Q[4]+Q[3]*Q[2])*invdet, -( Q[0]*Q[4]-Q[1]*Q[2])*invdet,  ( Q[0]*Q[3]-Q[1]*Q[1])*invdet}
    };
   const double dPP[3][3] = {
        {gradQin[55][0],gradQin[56][0],gradQin[57][0]},
        {gradQin[55][1],gradQin[56][1],gradQin[57][1]},
        {gradQin[55][2],gradQin[56][2],gradQin[57][2]}
    };

    // NOTE Aex is symmetric
    double Aex[3][3] = { {Q[6], Q[7], Q[8]}, {Q[7], Q[9], Q[10]}, {Q[8], Q[10], Q[11]} };

    double trAex = 0;
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) trAex+=g_contr[i][j]*Aex[i][j];
    //trAex *= 1./3;

    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Aex[i][j] -= 1./3. * trAex * g_cov[i][j];

    // Matrix multiplications Amix = matmul(g_contr, Aex) Aup  = matmul(g_contr, mytranspose(Amix))
    double Amix[3][3]={0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int u = 0; u < 3; u++) Amix[i][j] += g_contr[i][u] * Aex[u][j];
    double Aup[3][3]={0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int u = 0; u < 3; u++) Aup[i][j] += g_contr[i][u] * Amix[j][u]; // Note the transposition is in the indices

    const double DD[3][3][3] = {
        {{Q[37], Q[38], Q[39]}, {Q[38], Q[40], Q[41]}, {Q[39], Q[41], Q[42]}},
        {{Q[43], Q[44], Q[45]}, {Q[44], Q[46], Q[47]}, {Q[45], Q[47], Q[48]}},
        {{Q[49], Q[50], Q[51]}, {Q[50], Q[52], Q[53]}, {Q[51], Q[53], Q[54]}}
    };
    double dgup[3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int k = 0; k < 3; k++)
    for (int m = 0; m < 3; m++)
    for (int l = 0; l < 3; l++)
    for (int n = 0; n < 3; n++)
    for (int j = 0; j < 3; j++) dgup[k][m][l] -= g_contr[m][n]*g_contr[j][l]*2*DD[k][n][j];

    const double PP[3] = {Q[55], Q[56], Q[57]};

    double Christoffel[3][3][3]       = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double Christoffel_tilde[3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    //double Christoffel_kind1[3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    {
        //Christoffel_kind1[i][j][k] = DD[k][i][j] + DD[j][i][k] - DD[i][j][k];
        for (int l = 0; l < 3; l++)
        {
            Christoffel_tilde[i][j][k] += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] );
            Christoffel[i][j][k]       += g_contr[k][l] * ( DD[i][j][l] + DD[j][i][l] - DD[l][i][j] ) - g_contr[k][l] * ( g_cov[j][l] * PP[i] + g_cov[i][l] * PP[j] - g_cov[i][j] * PP[l] );
        }
    }

    double Gtilde[3] = {0,0,0};
    for (int l = 0; l < 3; l++)
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) Gtilde[i] += g_contr[j][l] * Christoffel_tilde[j][l][i];

    const double Ghat[3] = {Q[13], Q[14], Q[15]};

    const double phi = std::exp(std::fmax(-20., std::fmin(20.,Q[24])));
    const double phi2 = phi*phi;

    double Z[3] = {0,0,0};
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Z[i] += 0.5*( g_cov[i][j]* (Ghat[j] - Gtilde[j]));
    double Zup[3] = {0,0,0};
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++) Zup[i] += phi2 * g_contr[i][j] * Z[j];


    const double dDD[3][3][3][3] = {
        {
            {
                {gradQin[37][0],gradQin[38][0],gradQin[39][0]}, {gradQin[38][0],gradQin[40][0],gradQin[41][0]}, {gradQin[39][0],gradQin[41][0],gradQin[42][0]},
            },
            {
                {gradQin[43][0],gradQin[44][0],gradQin[45][0]}, {gradQin[44][0],gradQin[46][0],gradQin[47][0]}, {gradQin[45][0],gradQin[47][0],gradQin[48][0]},
            },
            {
                {gradQin[49][0],gradQin[50][0],gradQin[51][0]}, {gradQin[50][0],gradQin[52][0],gradQin[53][0]}, {gradQin[51][0],gradQin[53][0],gradQin[54][0]}
            }
        },
        {
            {
                {gradQin[37][1],gradQin[38][1],gradQin[39][1]}, {gradQin[38][1],gradQin[40][1],gradQin[41][1]}, {gradQin[39][1],gradQin[41][1],gradQin[42][1]},
            },
            {
                {gradQin[43][1],gradQin[44][1],gradQin[45][1]}, {gradQin[44][1],gradQin[46][1],gradQin[47][1]}, {gradQin[45][1],gradQin[47][1],gradQin[48][1]},
            },
            {
                {gradQin[49][1],gradQin[50][1],gradQin[51][1]}, {gradQin[50][1],gradQin[52][1],gradQin[53][1]}, {gradQin[51][1],gradQin[53][1],gradQin[54][1]}
            }
        },
        {
            {
                {gradQin[37][2],gradQin[38][2],gradQin[39][2]}, {gradQin[38][2],gradQin[40][2],gradQin[41][2]}, {gradQin[39][2],gradQin[41][2],gradQin[42][2]},
            },
            {
                {gradQin[43][2],gradQin[44][2],gradQin[45][2]}, {gradQin[44][2],gradQin[46][2],gradQin[47][2]}, {gradQin[45][2],gradQin[47][2],gradQin[48][2]},
            },
            {
                {gradQin[49][2],gradQin[50][2],gradQin[51][2]}, {gradQin[50][2],gradQin[52][2],gradQin[53][2]}, {gradQin[51][2],gradQin[53][2],gradQin[54][2]}
            }
        }
    };


    double dChristoffelNCP[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    double dChristoffel_tildeNCP[3][3][3][3] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++)
    {
        dChristoffelNCP      [k][i][j][m] = 0;
        dChristoffel_tildeNCP[k][i][j][m] = 0;
        for (int l = 0; l < 3; l++)
        {
            dChristoffelNCP[k][i][j][m] +=  0.5*g_contr[m][l] * (
                    dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]
                    - g_cov[j][l]*(dPP[k][i] + dPP[i][k]) - g_cov[i][l]*(dPP[k][j]+dPP[j][k]) +  g_cov[i][j]*(dPP[k][l]+dPP[l][k]) );

            dChristoffel_tildeNCP[k][i][j][m] += 0.5*g_contr[m][l]*(dDD[k][i][j][l] + dDD[i][k][j][l] + dDD[k][j][i][l] + dDD[j][k][i][l] - dDD[k][l][i][j] - dDD[l][k][i][j]);

        }
    }

    double RiemannNCP[3][3][3][3] = {0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++) RiemannNCP[i][k][j][m] = dChristoffelNCP[k][i][j][m] - dChristoffelNCP[j][i][k][m];

    double RicciNCP[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++)
    for (int l = 0; l < 3; l++) RicciNCP[m][n] += RiemannNCP[m][l][n][l];

    double dGtildeNCP[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
    for (int j = 0; j < 3; j++)
    for (int l = 0; l < 3; l++) dGtildeNCP[k][i] += g_contr[j][l]*dChristoffel_tildeNCP[k][j][l][i];


    const double dGhat[3][3] = {
        {gradQin[13][0],gradQin[14][0],gradQin[15][0]},
        {gradQin[13][1],gradQin[14][1],gradQin[15][1]},
        {gradQin[13][2],gradQin[14][2],gradQin[15][2]}
    };

    double dZNCP[3][3] = {0,0,0,0,0,0,0,0,0};
    for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++) dZNCP[k][i] += 0.5*g_cov[i][j]*(dGhat[k][j]-dGtildeNCP[k][j]);

    double RicciPlusNablaZNCP[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) RicciPlusNablaZNCP[i][j] = RicciNCP[i][j] + dZNCP[i][j] + dZNCP[j][i];

    double RPlusTwoNablaZNCP = 0;
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) RPlusTwoNablaZNCP += g_contr[i][j]*RicciPlusNablaZNCP[i][j];
    RPlusTwoNablaZNCP*=phi2;

    NCPterm[0]  =   RPlusTwoNablaZNCP;
    //if (NCPterm>0.0) std::cout<< NCPterm << std::endl; 
}

void exagrype::ccz4::maxEigenvalue(
        const double* const Q,
        int                 normal,
        const double        CCZ4e,
        const double        CCZ4GLMc,
        const double        CCZ4GLMd,
        double*             maxEigenvalue
) {
  const double np = 1.0;
  const double qmin = std::min({Q[0],Q[3],Q[5]});
  const double alpha = std::max({1.0, Q[16]}) * std::max({1.0, pow(Q[24],1.0/np)})/std::sqrt(qmin);

  constexpr double sqrtwo = 1.4142135623730951;
  const double tempA = alpha * std::max({sqrtwo, CCZ4e, CCZ4GLMc/alpha, CCZ4GLMd/alpha});
  const double tempB = Q[17+normal];//DOT_PRODUCT(Q(18:20),nv(:))
  maxEigenvalue[0] = std::max({1.0, std::abs(-tempA-tempB), std::abs(tempA-tempB)});
}

//this version use alpha and phi
template <bool SecondOrder>
static void sourceImplementation(double* S, double* Saux, const double* const Q, const double* const Qaux,
      const int CCZ4LapseType,
      const double CCZ4c,
      const double CCZ4e,
      const double CCZ4f,
      const int CCZ4b,
      const int CCZ4s,
      const double CCZ4itau,
      const double CCZ4eta,
      const double CCZ4k1,
      const double CCZ4k2,
      const double CCZ4k3
      )
{
    // Vacuum CCZ4: 58 entries, with symmetric tensor contractions.
    // 1 in the first-order formulation: enables the advection terms which are
    // written in the secondary entries, and the trace constraint terms of Ghat.
    constexpr int CCZ4FO = SecondOrder ? 0 : 1;

    // Preserve the upstream lapse floor and alpha (not log(alpha)) variables.
    const double alpha = std::fmax(1e-4,Q[16]);

    // Match CCZ4PlusScalarKernels: 0 = harmonic, 1 = 1+log,
    // all other values disable the slicing driver (advection is retained).
    double ga  = 0.0;
    double gaa = 0.0;
    if (CCZ4LapseType == 0)
    {
      ga = 1.0;
    }
    else if (CCZ4LapseType == 1)
    {
      ga  =  2./alpha;
      gaa = -ga/alpha;
    }

    const double np = 1.0;
    const int CCZ4Sa = 0;
    const int CCZ4Sp = 0;

    const double g_cov[3][3] = { {Q[0], Q[1], Q[2]},
                                 {Q[1], Q[3], Q[4]},
                                 {Q[2], Q[4], Q[5]} };
    const double det = Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] - Q[2]*Q[2]*Q[3];
    const double invdet = 1./det;
    const double g_contr_12 = -( Q[1]*Q[5]-Q[2]*Q[4])*invdet;
    const double g_contr_13 = -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet;
    const double g_contr_23 = -( Q[0]*Q[4]-Q[2]*Q[1])*invdet;
    const double g_contr[3][3] = {{(Q[3]*Q[5]-Q[4]*Q[4])*invdet, g_contr_12, g_contr_13},
                                  {g_contr_12, (Q[0]*Q[5]-Q[2]*Q[2])*invdet, g_contr_23},
                                  {g_contr_13, g_contr_23, (Q[0]*Q[3]-Q[1]*Q[1])*invdet}};

    double Aex[3][3] = { {Q[6], Q[7], Q[8]},
                         {Q[7], Q[9], Q[10]},
                         {Q[8], Q[10], Q[11]} };

    const double trAex = g_contr[0][0] * Aex[0][0]
                       + g_contr[1][1] * Aex[1][1]
                       + g_contr[2][2] * Aex[2][2]
                       + 2.0 * ( g_contr[0][1] * Aex[0][1]
                               + g_contr[0][2] * Aex[0][2]
                               + g_contr[1][2] * Aex[1][2] );

    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        Aex[i][j] -= 1./3. * trAex * g_cov[i][j];
        Aex[j][i] = Aex[i][j];
    }

    double Amix[3][3] = {0};
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    for (int u = 0; u < 3; u++) Amix[i][j] += g_contr[i][u] * Aex[u][j];

    double Aup[3][3] = {0};
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        for (int u = 0; u < 3; u++) Aup[i][j] += g_contr[i][u] * Amix[j][u];
        Aup[j][i] = Aup[i][j];
    }

    const double Theta = Q[12];

    const double DD[3][3][3] = {{ {Qaux[12], Qaux[13], Qaux[14]},
                                  {Qaux[13], Qaux[15], Qaux[16]},
                                  {Qaux[14], Qaux[16], Qaux[17]} },
                                { {Qaux[18], Qaux[19], Qaux[20]},
                                  {Qaux[19], Qaux[21], Qaux[22]},
                                  {Qaux[20], Qaux[22], Qaux[23]} },
                                { {Qaux[24], Qaux[25], Qaux[26]},
                                  {Qaux[25], Qaux[27], Qaux[28]},
                                  {Qaux[26], Qaux[28], Qaux[29]} }};

    double DDup[3][3][3] = {0};
    for (int k = 0; k < 3; k++)
    for (int m = 0; m < 3; m++)
    for (int l = m; l < 3; l++)
    {
        for (int n = 0; n < 3; n++)
        for (int j = 0; j < 3; j++) DDup[k][m][l] += g_contr[m][n] * g_contr[l][j] * DD[k][n][j];

    }

    const double phi = std::fmax(1e-2,pow(Q[24],(1.0/np)));
    const double phi2 = phi*phi;
    const double chi = pow(phi,np);

    const double PP[3] = {Qaux[30], Qaux[31], Qaux[32]};

    double Christoffel_tilde_down[3][3][3] = {0};
    double Christoffel_tilde[3][3][3] = {0};
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            Christoffel_tilde_down[k][i][j] = DD[i][j][k] + DD[j][i][k] - DD[k][i][j];
            Christoffel_tilde_down[k][j][i] = Christoffel_tilde_down[k][i][j];
        }
        for (int k = 0; k < 3; k++)
        {
            for (int l = 0; l < 3; l++) Christoffel_tilde[i][j][k] += g_contr[k][l] * Christoffel_tilde_down[l][i][j];
            Christoffel_tilde[j][i][k] = Christoffel_tilde[i][j][k];
        }
    }

    double Gtilde[3] = {0};
    for (int i = 0; i < 3; i++)
    {
        Gtilde[i] = g_contr[0][0] * Christoffel_tilde[0][0][i]
                  + g_contr[1][1] * Christoffel_tilde[1][1][i]
                  + g_contr[2][2] * Christoffel_tilde[2][2][i]
                  + 2.0 * ( g_contr[0][1] * Christoffel_tilde[0][1][i]
                          + g_contr[0][2] * Christoffel_tilde[0][2][i]
                          + g_contr[1][2] * Christoffel_tilde[1][2][i] );
    }

    const double Ghat[3] = {Q[13], Q[14], Q[15]};

    double DeltaGamma[3];
    for (int i = 0; i < 3; i++) DeltaGamma[i] = Ghat[i] - Gtilde[i];

    double Zup[3] = {0};
    for (int i = 0; i < 3; i++) Zup[i] = 0.5 * phi2 * DeltaGamma[i];

    const double PPdotPP = g_contr[0][0] * PP[0] * PP[0]
                         + g_contr[1][1] * PP[1] * PP[1]
                         + g_contr[2][2] * PP[2] * PP[2]
                         + 2.0 * ( g_contr[0][1] * PP[0] * PP[1]
                                 + g_contr[0][2] * PP[0] * PP[2]
                                 + g_contr[1][2] * PP[1] * PP[2] );

    double PPdotGtilde = 0;
    for (int k = 0; k < 3; k++) PPdotGtilde += PP[k] * Gtilde[k];

    const double AA[3] = {Qaux[0], Qaux[1], Qaux[2]};

    double AAdotPP = 0;
    for (int k = 0; k < 3; k++)
    for (int l = 0; l < 3; l++) AAdotPP += g_contr[k][l] * AA[k] * PP[l];

    const double Va = (CCZ4Sa == 0) ? 1.0 : alpha;
    const double Vp = (CCZ4Sp == 0) ? 1.0/np/chi : 1.0;

    double Riccihat[3][3];
    const double Vp2 = Vp * Vp;
    const double Riccihatfactor1 = CCZ4Sp - (np - 1.0) * Vp2;
    const double Riccihatfactor2 = CCZ4Sp - (np + 1.0) * Vp2;

    for (int i = 0; i < 3; i++)

    for (int j = i; j < 3; j++)
    {
        double temp1=0, temp2=0, temp3=0, temp4=0;
        for (int k = 0; k < 3; k++)
        {
            temp1 += Ghat[k] * DD[k][i][j];
            temp2 += PP[k] * Christoffel_tilde[i][j][k];
            temp3 += DeltaGamma[k] * ( g_cov[i][k] * PP[j]
                                     + g_cov[j][k] * PP[i]
                                     - g_cov[i][j] * PP[k] );
            for (int l = 0; l < 3; l++)
            for (int m = 0; m < 3; m++)
            {
                temp4 += g_contr[l][m] * ( Christoffel_tilde[l][i][k] * Christoffel_tilde_down[j][k][m]
                                         + Christoffel_tilde[l][j][k] * Christoffel_tilde_down[i][k][m]
                                         + Christoffel_tilde[i][m][k] * Christoffel_tilde_down[k][l][j] );
            }
        }
        Riccihat[i][j] = Riccihatfactor1 * PP[i] * PP[j]
                       + temp1 - Vp * temp2 + Vp * temp3 + temp4
                       + (Riccihatfactor2 * PPdotPP - Vp * PPdotGtilde) * g_cov[i][j];
        Riccihat[j][i] = Riccihat[i][j];
    }

    double trRiccihat = g_contr[0][0] * Riccihat[0][0]
                      + g_contr[1][1] * Riccihat[1][1]
                      + g_contr[2][2] * Riccihat[2][2]
                      + 2.0 * ( g_contr[0][1] * Riccihat[0][1]
                              + g_contr[0][2] * Riccihat[0][2]
                              + g_contr[1][2] * Riccihat[1][2] );
    double TrRiccihat = phi2 * trRiccihat;

    double DDalpha[3][3] = {0};

    for (int i = 0; i < 3; i++)

    for (int j = i; j < 3; j++)
    {

        double temp = 0;
        for (int k = 0; k < 3; k++) temp += Christoffel_tilde[i][j][k] * AA[k];
        DDalpha[i][j] = Va * Vp * (PP[i] * AA[j] + PP[j] * AA[i] - AAdotPP * g_cov[i][j])
                      + CCZ4Sa * alpha * AA[i] * AA[j]
                      - Va * temp;
        DDalpha[j][i] = DDalpha[i][j];

    }

    double trDDalpha = g_contr[0][0] * DDalpha[0][0]
                     + g_contr[1][1] * DDalpha[1][1]
                     + g_contr[2][2] * DDalpha[2][2]
                     + 2.0 * ( g_contr[0][1] * DDalpha[0][1]
                             + g_contr[0][2] * DDalpha[0][2]
                             + g_contr[1][2] * DDalpha[1][2] );
    double TrDDalpha = phi2 * trDDalpha;

    double SecondOrderTerms[3][3];

    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        SecondOrderTerms[i][j] = alpha * Riccihat[i][j] - DDalpha[i][j];
        SecondOrderTerms[j][i] = SecondOrderTerms[i][j];
    }

    double trSecondOrderTerms = alpha * trRiccihat - trDDalpha;

    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        SecondOrderTerms[i][j] -= 1./3 * trSecondOrderTerms * g_cov[i][j];
        SecondOrderTerms[j][i] = SecondOrderTerms[i][j];
    }

    const double BB[3][3] = { {Qaux[3], Qaux[4], Qaux[5]},
                              {Qaux[6], Qaux[7], Qaux[8]},
                              {Qaux[9], Qaux[10], Qaux[11]} };

    double TrBB = BB[0][0] + BB[1][1] + BB[2][2];
    const double beta[3] = {Q[17], Q[18], Q[19]};

    double dtgamma[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        dtgamma[i][j] = -2.0 * alpha * Aex[i][j]
                      - CCZ4itau * (det - 1.0) * g_cov[i][j]
                      - 2.0 / 3.0 * g_cov[i][j] * TrBB;
        double temp = 0;
        for (int k = 0; k < 3; k++)
        {
            dtgamma[i][j] += g_cov[k][i] * BB[j][k] + g_cov[k][j] * BB[i][k];
            temp += beta[k] * DD[k][i][j];
        }
        dtgamma[i][j] += CCZ4FO * 2.0 * temp;
    }

    const double Kex = Q[23];

    double dtAex[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        dtAex[i][j] = phi2 * SecondOrderTerms[i][j]
                    + alpha * Aex[i][j] * (Kex - 2. * Theta * CCZ4c)
                    - 2./3. * Aex[i][j] * TrBB;
        for (int k = 0; k < 3; k++)
        {
            dtAex[i][j] +=  -2. * alpha * Aex[i][k] * Amix[k][j]
                         + Aex[k][i] * BB[j][k]
                         + Aex[k][j] * BB[i][k];
        }
    }


    // k1 is already lapse-rescaled: retain k1, not alpha*k1, in damping.
    double dtKex = -TrDDalpha
                 + alpha * (TrRiccihat + Kex * Kex - 2.0 * CCZ4c * Theta * Kex)
                 - 3.0 * CCZ4k1 * (1.0 + CCZ4k2) * Theta;

    double dtchi = CCZ4FO * ((CCZ4Sp == 0) ? 1.0 : phi) * (beta[0]*PP[0]+beta[1]*PP[1]+beta[2]*PP[2])
                 + np / 3. * (alpha * Kex - TrBB) * chi;

    double dtalpha = - alpha * alpha * ga * (Kex - 2. * CCZ4c * Theta)
                   + CCZ4FO * Va * (beta[0]*AA[0]+beta[1]*AA[1]+beta[2]*AA[2]);

    const double AupAdown = Aex[0][0] * Aup[0][0]
                          + Aex[1][1] * Aup[1][1]
                          + Aex[2][2] * Aup[2][2]
                          + 2.0 * ( Aex[0][1] * Aup[0][1]
                                  + Aex[0][2] * Aup[0][2]
                                  + Aex[1][2] * Aup[1][2] );

    double dtTheta = 0.5 * alpha * CCZ4e * CCZ4e * ( TrRiccihat - AupAdown + 2./3. * Kex * Kex )
                   - alpha * CCZ4c * Theta * Kex
                   - CCZ4k1 * (2. + CCZ4k2) * Theta
                   - Va * (Zup[0]*AA[0]+Zup[1]*AA[1]+Zup[2]*AA[2]);

    // II_k = 2 alpha D_k^{ij} A_ij; only upper-triangular DDup is read.
    double II[3];
    for (int i = 0; i < 3; i++)
    {
        II[i] = 2.0 * alpha * ( DDup[i][0][0] * Aex[0][0]
                              + DDup[i][1][1] * Aex[1][1]
                              + DDup[i][2][2] * Aex[2][2]
                              + 2.0 * ( DDup[i][0][1] * Aex[0][1]
                                      + DDup[i][0][2] * Aex[0][2]
                                     + DDup[i][1][2] * Aex[1][2] ) );
    }

    double dtGhat[3];
    for (int i = 0; i < 3; i++)
    {
        double temp1=0, temp2=0, temp3=0, temp4=0, temp5=0, temp7=0;
        for (int j = 0; j < 3; j++)
        {
            temp1 += Gtilde[j] * BB[j][i];
            temp2 += g_contr[i][j] * AA[j];
            temp3 += DeltaGamma[j] * BB[j][i];
            temp4 += Aup[i][j] * AA[j];
            temp5 += Aup[i][j] * PP[j];
            temp7 += g_contr[i][j] * II[j];
        }
        const double temp6 = Christoffel_tilde[0][0][i] * Aup[0][0]
                           + Christoffel_tilde[1][1][i] * Aup[1][1]
                           + Christoffel_tilde[2][2][i] * Aup[2][2]
                           + 2.0 * ( Christoffel_tilde[0][1][i] * Aup[0][1]
                                   + Christoffel_tilde[0][2][i] * Aup[0][2]
                                   + Christoffel_tilde[1][2][i] * Aup[1][2] );

        dtGhat[i] = 2./3. * TrBB * ( Gtilde[i] + CCZ4k3 * DeltaGamma[i] )
                  - ( 2./3. * alpha * Kex + CCZ4k1 ) * DeltaGamma[i]
                  - temp1
                  - 2. * Va * Theta * temp2
                  - CCZ4k3 * temp3
                  - 2. * Va * temp4
                  - 6. * alpha * Vp * temp5
                  + 2. * alpha * temp6
                  - 2. * CCZ4FO * CCZ4s * temp7;
    }

    const double myb[3] = {Q[20], Q[21], Q[22]};

    double dtmyb[3];
    for (int i = 0; i < 3; i++) dtmyb[i] = CCZ4s * ( dtGhat[i] - CCZ4eta * myb[i] );

    // CCZ4s controls the shift driver; CCZ4b its advection terms (Kernels.h).
    // SO primary-variable advection is handled outside this source term.
    double dtbeta[3];

    for (int i = 0; i < 3; i++)
    {
        dtbeta[i] = CCZ4f * myb[i]
                   + CCZ4b * CCZ4FO * (beta[0]*BB[0][i]+beta[1]*BB[1][i]+beta[2]*BB[2][i]);

        dtbeta[i] *= CCZ4s;
    }

    S[0]  = dtgamma[0][0];
    S[1]  = dtgamma[0][1];
    S[2]  = dtgamma[0][2];
    S[3]  = dtgamma[1][1];
    S[4]  = dtgamma[1][2];
    S[5]  = dtgamma[2][2];
    S[6]  = dtAex[0][0];
    S[7]  = dtAex[0][1];
    S[8]  = dtAex[0][2];
    S[9]  = dtAex[1][1];
    S[10] = dtAex[1][2];
    S[11] = dtAex[2][2];

    S[12] = dtTheta;
    for (int i = 0; i < 3; i++) S[13+i] = dtGhat[i];
    S[16] = dtalpha;
    for (int i = 0; i < 3; i++) S[17+i] = dtbeta[i];
    for (int i = 0; i < 3; i++) S[20+i] = dtmyb[i];
    S[23] = dtKex;
    S[24] = dtchi;

    // The secondary entries. In the second-order formulation, they are not
    // evolved, and the caller may omit the pointer.
    if constexpr (!SecondOrder) {
        double dtAA[3];
        double iVa = (CCZ4Sa == 0) ? alpha : 1.0;

        for (int i = 0; i < 3; i++)
        {

          dtAA[i] = - alpha * AA[i] * ( (2. - CCZ4Sa) * ga + alpha * gaa ) * ( Kex - 2. * CCZ4c * Theta )
                 + CCZ4s * iVa * ga * II[i]
                 + BB[i][0]*AA[0]+BB[i][1]*AA[1]+BB[i][2]*AA[2];
        }

        double dtBB[3][3];

        for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
        {
            double temp = 0.0;
            for (int k = 0; k < 3; k++)
                temp += BB[j][k] * BB[k][i];
            // Derivative of the advection term beta^k B_k^i of the shift equation
            dtBB[j][i] = CCZ4s * CCZ4b * temp;
        }

        double dtDD[3][3][3];

        for (int k = 0; k < 3; k++)
        for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++)
        {
            double temp = 0;
            for (int l = 0; l < 3; l++)
            {
                temp += DD[l][i][j] * BB[k][l]
                      + DD[k][l][i] * BB[j][l]
                      + DD[k][l][j] * BB[i][l];
            }
            dtDD[k][i][j] = - Va * AA[k] * Aex[i][j]
                          - 2./3. * TrBB * DD[k][i][j]
                          - 1./3. * g_cov[i][j] * II[k]
                          + temp;

        }

        double dtPP[3];
        for (int i = 0; i < 3; i++)
        {
            double temp = 0;
            for (int j = 0; j < 3; j++) temp += BB[i][j] * PP[j];
            dtPP[i] = 1./3. / Vp * ( Va * Kex * AA[i] - CCZ4s * II[i] )
                    + (1 - CCZ4Sp) * np/3. * (alpha * Kex - TrBB) * PP[i]
                    + temp;
        }

        for (int i = 0; i < 3; i++) Saux[i] = dtAA[i];
        Saux[3] = dtBB[0][0];
        Saux[4] = dtBB[0][1];
        Saux[5] = dtBB[0][2];
        Saux[6] = dtBB[1][0];
        Saux[7] = dtBB[1][1];
        Saux[8] = dtBB[1][2];
        Saux[9] = dtBB[2][0];
        Saux[10] = dtBB[2][1];
        Saux[11] = dtBB[2][2];
        Saux[12] = dtDD[0][0][0];
        Saux[13] = dtDD[0][0][1];
        Saux[14] = dtDD[0][0][2];
        Saux[15] = dtDD[0][1][1];
        Saux[16] = dtDD[0][1][2];
        Saux[17] = dtDD[0][2][2];
        Saux[18] = dtDD[1][0][0];
        Saux[19] = dtDD[1][0][1];
        Saux[20] = dtDD[1][0][2];
        Saux[21] = dtDD[1][1][1];
        Saux[22] = dtDD[1][1][2];
        Saux[23] = dtDD[1][2][2];
        Saux[24] = dtDD[2][0][0];
        Saux[25] = dtDD[2][0][1];
        Saux[26] = dtDD[2][0][2];
        Saux[27] = dtDD[2][1][1];
        Saux[28] = dtDD[2][1][2];
        Saux[29] = dtDD[2][2][2];
        for (int i = 0; i < 3; i++) Saux[30+i] = dtPP[i];
    }
}

template <bool SecondOrder, bool Advection>
static void ncpImplementation(double* BgradQ, double* BgradQaux, const double* const Q, const double* const Qaux, const double* const gradQ, const double* const gradQaux, const int normal,
      const int CCZ4LapseType,
      const double CCZ4c,
      const double CCZ4e,
      const double CCZ4f,
      const int CCZ4b,
      const int CCZ4s,
      const double CCZ4mu
      )
{
    // 1 in the first-order formulation: enables the advection terms which are
    // written in the secondary entries, and the trace constraint terms of Ghat.
    constexpr int CCZ4FO = SecondOrder ? 0 : 1;

    // The advection terms beta^k d_k Q are compiled only if Advection is set.
    // The kernels below do not set it: the numerical scheme adds these terms
    // with lopsided stencils, and they must not be counted twice.

    const double alpha = std::fmax(1e-4,Q[16]);

    const double alpha2 = alpha * alpha;
    // Same slicing-driver mapping as source() and CCZ4PlusScalarKernels.
    double ga  = 0.0;
    if (CCZ4LapseType == 0)
    {
      ga = 1.0;
    }
    else if (CCZ4LapseType == 1)
    {
      ga  =  2./alpha;
    }

    const double np = 1.0;
    const int CCZ4Sa = 0;
    const int CCZ4Sp = 0;

    double gradQin[58][3] = {0};

    // Only the derivative along the normal is given.
    for (int i = 0; i < 25; i++) gradQin[i][normal]    = gradQ[i];
    for (int i = 0; i < 33; i++) gradQin[25+i][normal] = gradQaux[i];

    const double g_cov[3][3] = { {Q[0], Q[1], Q[2]},
                                 {Q[1], Q[3], Q[4]},
                                 {Q[2], Q[4], Q[5]} };
    const double invdet = 1./(Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] - Q[2]*Q[2]*Q[3]);

    const double g_contr_12 = -( Q[1]*Q[5]-Q[2]*Q[4])*invdet;
    const double g_contr_13 = -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet;
    const double g_contr_23 = -( Q[0]*Q[4]-Q[2]*Q[1])*invdet;
    const double g_contr[3][3] = {{(Q[3]*Q[5]-Q[4]*Q[4])*invdet, g_contr_12, g_contr_13},
                                  {g_contr_12, (Q[0]*Q[5]-Q[2]*Q[2])*invdet, g_contr_23},
                                  {g_contr_13, g_contr_23, (Q[0]*Q[3]-Q[1]*Q[1])*invdet}};

    const double phi = std::fmax(1e-2,pow(Q[24],1.0/np));
    const double chi  = pow(phi,np);
    const double phi2 = phi*phi;

    double dPP[3][3] = { {gradQin[55][0],gradQin[56][0],gradQin[57][0]},
                         {gradQin[55][1],gradQin[56][1],gradQin[57][1]},
                         {gradQin[55][2],gradQin[56][2],gradQin[57][2]} };

    double dAA[3][3] = { {gradQin[25][0],gradQin[26][0],gradQin[27][0]},
                         {gradQin[25][1],gradQin[26][1],gradQin[27][1]},
                         {gradQin[25][2],gradQin[26][2],gradQin[27][2]} };

    const double dDD[3][3][3][3] = {
        {
            {
                {gradQin[37][0],gradQin[38][0],gradQin[39][0]}, {gradQin[38][0],gradQin[40][0],gradQin[41][0]}, {gradQin[39][0],gradQin[41][0],gradQin[42][0]},
            },
            {
                {gradQin[43][0],gradQin[44][0],gradQin[45][0]}, {gradQin[44][0],gradQin[46][0],gradQin[47][0]}, {gradQin[45][0],gradQin[47][0],gradQin[48][0]},
            },
            {
                {gradQin[49][0],gradQin[50][0],gradQin[51][0]}, {gradQin[50][0],gradQin[52][0],gradQin[53][0]}, {gradQin[51][0],gradQin[53][0],gradQin[54][0]}
            }
        },
        {
            {
                {gradQin[37][1],gradQin[38][1],gradQin[39][1]}, {gradQin[38][1],gradQin[40][1],gradQin[41][1]}, {gradQin[39][1],gradQin[41][1],gradQin[42][1]},
            },
            {
                {gradQin[43][1],gradQin[44][1],gradQin[45][1]}, {gradQin[44][1],gradQin[46][1],gradQin[47][1]}, {gradQin[45][1],gradQin[47][1],gradQin[48][1]},
            },
            {
                {gradQin[49][1],gradQin[50][1],gradQin[51][1]}, {gradQin[50][1],gradQin[52][1],gradQin[53][1]}, {gradQin[51][1],gradQin[53][1],gradQin[54][1]}
            }
        },
        {
            {
                {gradQin[37][2],gradQin[38][2],gradQin[39][2]}, {gradQin[38][2],gradQin[40][2],gradQin[41][2]}, {gradQin[39][2],gradQin[41][2],gradQin[42][2]},
            },
            {
                {gradQin[43][2],gradQin[44][2],gradQin[45][2]}, {gradQin[44][2],gradQin[46][2],gradQin[47][2]}, {gradQin[45][2],gradQin[47][2],gradQin[48][2]},
            },
            {
                {gradQin[49][2],gradQin[50][2],gradQin[51][2]}, {gradQin[50][2],gradQin[52][2],gradQin[53][2]}, {gradQin[51][2],gradQin[53][2],gradQin[54][2]}
            }
        }
    };

    const double dGhat[3][3] = {
        {gradQin[13][0],gradQin[14][0],gradQin[15][0]},
        {gradQin[13][1],gradQin[14][1],gradQin[15][1]},
        {gradQin[13][2],gradQin[14][2],gradQin[15][2]}
    };

    double Vp = (CCZ4Sp == 0) ? 1.0/np/chi : 1.0;

    const double trdPP = g_contr[0][0] * dPP[0][0]
                       + g_contr[1][1] * dPP[1][1]
                       + g_contr[2][2] * dPP[2][2]
                       + g_contr[0][1] * (dPP[0][1] + dPP[1][0])
                       + g_contr[0][2] * (dPP[0][2] + dPP[2][0])
                       + g_contr[1][2] * (dPP[1][2] + dPP[2][1]);

    double Riccihat[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        double temp1 = g_cov[0][i] * dGhat[j][0] + g_cov[0][j] * dGhat[i][0]
                     + g_cov[1][i] * dGhat[j][1] + g_cov[1][j] * dGhat[i][1]
                     + g_cov[2][i] * dGhat[j][2] + g_cov[2][j] * dGhat[i][2];
        double temp2 = g_contr[0][0] * dDD[0][0][i][j]
                     + g_contr[1][1] * dDD[1][1][i][j]
                     + g_contr[2][2] * dDD[2][2][i][j]
                     + g_contr[0][1] * (dDD[0][1][i][j] + dDD[1][0][i][j])
                     + g_contr[0][2] * (dDD[0][2][i][j] + dDD[2][0][i][j])
                     + g_contr[1][2] * (dDD[1][2][i][j] + dDD[2][1][i][j]);
        Riccihat[i][j] = 0.5 * Vp * (dPP[i][j] + dPP[j][i])
                       + 0.5 * temp1 - temp2 + Vp * g_cov[i][j] * trdPP;
        Riccihat[j][i] = Riccihat[i][j];
    }

    double trRiccihat = g_contr[0][0] * Riccihat[0][0]
                      + g_contr[1][1] * Riccihat[1][1]
                      + g_contr[2][2] * Riccihat[2][2]
                      + 2.0 * ( g_contr[0][1] * Riccihat[0][1]
                              + g_contr[0][2] * Riccihat[0][2]
                              + g_contr[1][2] * Riccihat[1][2] );
    double TrRiccihat = phi2 * trRiccihat;

    double Va = (CCZ4Sa == 0) ? 1.0 : alpha;

    double DDalpha[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        DDalpha[i][j] = 0.5 * Va * ( dAA[i][j] + dAA[j][i] );
        DDalpha[j][i] = DDalpha[i][j];
    }

    double trDDalpha = g_contr[0][0] * DDalpha[0][0]
                     + g_contr[1][1] * DDalpha[1][1]
                     + g_contr[2][2] * DDalpha[2][2]
                     + 2.0 * ( g_contr[0][1] * DDalpha[0][1]
                             + g_contr[0][2] * DDalpha[0][2]
                             + g_contr[1][2] * DDalpha[1][2] );
    double TrDDalpha = phi2 * trDDalpha;

    double SecondOrderTerms[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        SecondOrderTerms[i][j] = - DDalpha[i][j] + alpha * Riccihat[i][j];
        SecondOrderTerms[j][i] = SecondOrderTerms[i][j];
    }

    double trSecondOrderTerms = alpha * trRiccihat - trDDalpha;

    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        SecondOrderTerms[i][j] -= 1./3 * trSecondOrderTerms * g_cov[i][j];
        SecondOrderTerms[j][i] = SecondOrderTerms[i][j];
    }

    const double beta[3] = {Q[17], Q[18], Q[19]};

    const double dAex[3][3][3] = {
        {{gradQin[6][0],gradQin[7][0],gradQin[8][0]}, {gradQin[7][0], gradQin[9][0], gradQin[10][0]},  {gradQin[8][0], gradQin[10][0], gradQin[11][0]}},
        {{gradQin[6][1],gradQin[7][1],gradQin[8][1]}, {gradQin[7][1], gradQin[9][1], gradQin[10][1]},  {gradQin[8][1], gradQin[10][1], gradQin[11][1]}},
        {{gradQin[6][2],gradQin[7][2],gradQin[8][2]}, {gradQin[7][2], gradQin[9][2], gradQin[10][2]},  {gradQin[8][2], gradQin[10][2], gradQin[11][2]}}
    };

    double dtAex[3][3];
    for (int i = 0; i < 3; i++)
    for (int j = i; j < 3; j++)
    {
        dtAex[i][j] = phi2 * SecondOrderTerms[i][j];
        if constexpr (Advection) dtAex[i][j] += beta[0]*dAex[0][i][j]+beta[1]*dAex[1][i][j]+beta[2]*dAex[2][i][j];
    }

    const double dKex[3] = {gradQin[23][0], gradQin[23][1], gradQin[23][2]};

    double dtKex = - TrDDalpha + alpha * TrRiccihat;
    if constexpr (Advection) dtKex += beta[0]*dKex[0]+beta[1]*dKex[1]+beta[2]*dKex[2];

    const double dTheta[3] = {gradQin[12][0],gradQin[12][1],gradQin[12][2]};
    double dtTheta = 0.5 * alpha * CCZ4e * CCZ4e * TrRiccihat;
    if constexpr (Advection) dtTheta += beta[0]*dTheta[0]+beta[1]*dTheta[1]+beta[2]*dTheta[2];

    const double dBB[3][3][3] = {
        {
            {gradQin[28][0],gradQin[29][0],gradQin[30][0]}, {gradQin[31][0],gradQin[32][0],gradQin[33][0]}, {gradQin[34][0],gradQin[35][0],gradQin[36][0]}
        },
        {
            {gradQin[28][1],gradQin[29][1],gradQin[30][1]}, {gradQin[31][1],gradQin[32][1],gradQin[33][1]}, {gradQin[34][1],gradQin[35][1],gradQin[36][1]}
        },
        {
            {gradQin[28][2],gradQin[29][2],gradQin[30][2]}, {gradQin[31][2],gradQin[32][2],gradQin[33][2]}, {gradQin[34][2],gradQin[35][2],gradQin[36][2]}
        }
    };

    double JJ[3];
    for (int i = 0; i < 3; i++)
    {
        JJ[i] = alpha * ( g_contr[0][0] * dAex[i][0][0]
                        + g_contr[1][1] * dAex[i][1][1]
                        + g_contr[2][2] * dAex[i][2][2]
                        + 2.0 * ( g_contr[0][1] * dAex[i][0][1]
                                + g_contr[0][2] * dAex[i][0][2]
                                + g_contr[1][2] * dAex[i][1][2] ) );
    }

    double dtGhat[3] = {0};
    for (int i = 0; i < 3; i++)
    {
        double temp1=0, temp2=0, temp3=0;
        for (int j = 0; j < 3; j++)
        {
            temp1 += g_contr[i][j] * ( alpha * (dTheta[j] - 2./3. * dKex[j]) + CCZ4FO * CCZ4s * JJ[j] );
            for (int k = 0; k < 3; k++)
            {
                temp2 += g_contr[j][k] * dBB[j][k][i];
                temp3 += g_contr[i][j] * ( dBB[j][k][k] + dBB[k][j][k] );
            }
        }
        dtGhat[i] = 2. * temp1 + CCZ4s * temp2 + CCZ4s * 1./6. * temp3;
        if constexpr (Advection) dtGhat[i] += beta[0]*dGhat[0][i]+beta[1]*dGhat[1][i]+beta[2]*dGhat[2][i];
    }

    double dtmyb[3];
    for (int i = 0; i < 3; i++)
    {
        // d_t b = s ( d_t Ghat - eta b + CCZ4b (beta^k d_k b - beta^k d_k Ghat) ),
        // where d_t Ghat comprises the advection of Ghat. Without Advection,
        // dtGhat lacks this advection, and the numerical scheme advects b
        // if CCZ4b is set. Hence, only the non-shifting shift is left to do.
        dtmyb[i] = dtGhat[i];
        if constexpr (Advection)
        {
            dtmyb[i] += CCZ4b * ( beta[0]*gradQin[20+i][0]+beta[1]*gradQin[20+i][1]+beta[2]*gradQin[20+i][2]
                                 - beta[0]*gradQin[13+i][0]-beta[1]*gradQin[13+i][1]-beta[2]*gradQin[13+i][2] );
        }
        else if (CCZ4b == 0)
        {
            dtmyb[i] += beta[0]*dGhat[0][i]+beta[1]*dGhat[1][i]+beta[2]*dGhat[2][i];
        }
        dtmyb[i] *= CCZ4s;
    }

    // Return B(Q) grad(Q), the negative of the derivative RHS computed above.
    BgradQ[0]  = 0.0;
    BgradQ[1]  = 0.0;
    BgradQ[2]  = 0.0;
    BgradQ[3]  = 0.0;
    BgradQ[4]  = 0.0;
    BgradQ[5]  = 0.0;
    BgradQ[6]  = -dtAex[0][0];
    BgradQ[7]  = -dtAex[0][1];
    BgradQ[8]  = -dtAex[0][2];
    BgradQ[9]  = -dtAex[1][1];
    BgradQ[10] = -dtAex[1][2];
    BgradQ[11] = -dtAex[2][2];

    BgradQ[12] = -dtTheta;
    for (int i = 0; i < 3; i++) BgradQ[13+i] = -dtGhat[i];
    BgradQ[16] = 0.0;
    for (int i = 0; i < 3; i++) BgradQ[17+i] = 0.0;
    for (int i = 0; i < 3; i++) BgradQ[20+i] = -dtmyb[i];



    BgradQ[23] = -dtKex;
    BgradQ[24] = 0.0;

    // The secondary entries. In the second-order formulation, they are not
    // evolved, and the caller may omit the pointer.
    if constexpr (!SecondOrder) {
        double dtAA[3] = {0};
        double iVa = (CCZ4Sa == 0) ? alpha : 1.0;

        for (int i = 0; i < 3; i++)
        {
            dtAA[i] = -alpha * iVa * ga * ( dKex[i] - 2. * CCZ4c * dTheta[i] )
                    - CCZ4s * iVa * ga * JJ[i];
            if constexpr (Advection) dtAA[i] += beta[0]*dAA[0][i] + beta[1]*dAA[1][i] + beta[2]*dAA[2][i];
        }

        double dtBB[3][3] = {{CCZ4f*gradQin[20][0],CCZ4f*gradQin[21][0],CCZ4f*gradQin[22][0]},
                             {CCZ4f*gradQin[20][1],CCZ4f*gradQin[21][1],CCZ4f*gradQin[22][1]},
                             {CCZ4f*gradQin[20][2],CCZ4f*gradQin[21][2],CCZ4f*gradQin[22][2]}};
        for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
        {
            double temp1=0, temp2=0;
            for (int k = 0; k < 3; k++)
            {
                temp1 += g_contr[i][k] * (dPP[j][k] - dPP[k][j]);
                for (int l = 0; l < 3; l++)
                for (int m = 0; m < 3; m++) temp2 += g_contr[i][k] * g_contr[l][m] * (dDD[j][m][k][l] - dDD[m][j][k][l]);
            }
            dtBB[j][i] += CCZ4mu * alpha2 * (Vp * temp1 - temp2);
            if constexpr (Advection) dtBB[j][i] += CCZ4b * (beta[0]*dBB[0][j][i]+beta[1]*dBB[1][j][i]+beta[2]*dBB[2][j][i]);
            dtBB[j][i] *= CCZ4s;
        }

        double dtDD[3][3][3];
        for (int k = 0; k < 3; k++)
        for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++)
        {
            double temp1=0, temp2=0;
            for (int l = 0; l < 3; l++)
            {
                temp1 += g_cov[i][j]* (dBB[k][l][l] + dBB[l][k][l]);
                temp2 += g_cov[l][i]* (dBB[k][j][l] + dBB[j][k][l]) + g_cov[l][j] * (dBB[k][i][l]+dBB[i][k][l]);
            }
            dtDD[k][i][j] = -alpha * dAex[k][i][j] + 1./3. * g_cov[i][j] * JJ[k]
                          - CCZ4s * 1./6. * temp1 + CCZ4s * 1./4. * temp2;
            if constexpr (Advection) dtDD[k][i][j] += beta[0]*dDD[0][k][i][j]+beta[1]*dDD[1][k][i][j]+beta[2]*dDD[2][k][i][j];
        }

        double dtPP[3];
        for (int i = 0; i < 3; i++)
        {
            double temp=0;
            for (int j = 0; j < 3; j++) temp += dBB[i][j][j] + dBB[j][i][j];
            dtPP[i] = 1./3./Vp * (alpha * dKex[i] + CCZ4s * JJ[i])
                    - CCZ4s * 1./6./Vp * temp;
            if constexpr (Advection) dtPP[i] += beta[0]*dPP[0][i]+beta[1]*dPP[1][i]+beta[2]*dPP[2][i];
        }

        for (int i = 0; i < 3; i++) BgradQaux[i] = -dtAA[i];
        BgradQaux[3] = -dtBB[0][0];
        BgradQaux[4] = -dtBB[0][1];
        BgradQaux[5] = -dtBB[0][2];
        BgradQaux[6] = -dtBB[1][0];
        BgradQaux[7] = -dtBB[1][1];
        BgradQaux[8] = -dtBB[1][2];
        BgradQaux[9] = -dtBB[2][0];
        BgradQaux[10] = -dtBB[2][1];
        BgradQaux[11] = -dtBB[2][2];
        BgradQaux[12] = -dtDD[0][0][0];
        BgradQaux[13] = -dtDD[0][0][1];
        BgradQaux[14] = -dtDD[0][0][2];
        BgradQaux[15] = -dtDD[0][1][1];
        BgradQaux[16] = -dtDD[0][1][2];
        BgradQaux[17] = -dtDD[0][2][2];
        BgradQaux[18] = -dtDD[1][0][0];
        BgradQaux[19] = -dtDD[1][0][1];
        BgradQaux[20] = -dtDD[1][0][2];
        BgradQaux[21] = -dtDD[1][1][1];
        BgradQaux[22] = -dtDD[1][1][2];
        BgradQaux[23] = -dtDD[1][2][2];
        BgradQaux[24] = -dtDD[2][0][0];
        BgradQaux[25] = -dtDD[2][0][1];
        BgradQaux[26] = -dtDD[2][0][2];
        BgradQaux[27] = -dtDD[2][1][1];
        BgradQaux[28] = -dtDD[2][1][2];
        BgradQaux[29] = -dtDD[2][2][2];
        for (int i = 0; i < 3; i++) BgradQaux[30+i] = -dtPP[i];
    }
}


void exagrype::ccz4::ncpFirstOrder(double* BgradQ, double* BgradQaux, const double* const Q, const double* const Qaux, const double* const gradQ, const double* const gradQaux, const int normal,
  const int CCZ4LapseType, const double CCZ4c, const double CCZ4e, const double CCZ4f, const int CCZ4b, const int CCZ4s, const double CCZ4mu
) {
  ncpImplementation<false, false>(BgradQ, BgradQaux, Q, Qaux, gradQ, gradQaux, normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4mu);
}


void exagrype::ccz4::ncpSecondOrder(double* BgradQ, const double* const Q, const double* const Qaux, const double* const gradQ, const double* const gradQaux, const int normal,
  const int CCZ4LapseType, const double CCZ4c, const double CCZ4e, const double CCZ4f, const int CCZ4b, const int CCZ4s, const double CCZ4mu
) {
  ncpImplementation<true, false>(BgradQ, nullptr, Q, Qaux, gradQ, gradQaux, normal, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4mu);
}


void exagrype::ccz4::sourceFirstOrder(double* S, double* Saux, const double* const Q, const double* const Qaux,
  const int CCZ4LapseType, const double CCZ4c, const double CCZ4e, const double CCZ4f, const int CCZ4b, const int CCZ4s, const double CCZ4itau, const double CCZ4eta, const double CCZ4k1, const double CCZ4k2, const double CCZ4k3
) {
  sourceImplementation<false>(S, Saux, Q, Qaux, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3);
}


void exagrype::ccz4::sourceSecondOrder(double* S, const double* const Q, const double* const Qaux,
  const int CCZ4LapseType, const double CCZ4c, const double CCZ4e, const double CCZ4f, const int CCZ4b, const int CCZ4s, const double CCZ4itau, const double CCZ4eta, const double CCZ4k1, const double CCZ4k2, const double CCZ4k3
) {
  sourceImplementation<true>(S, nullptr, Q, Qaux, CCZ4LapseType, CCZ4c, CCZ4e, CCZ4f, CCZ4b, CCZ4s, CCZ4itau, CCZ4eta, CCZ4k1, CCZ4k2, CCZ4k3);
}

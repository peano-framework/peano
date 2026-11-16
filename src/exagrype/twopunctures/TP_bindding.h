#pragma once

#include "exagrype/twopunctures/TwoPunctures.h"
//#include "AbstractFiniteVolumeCCZ4.h"
//The functions in this namespace works as binddings of two puncture library 

namespace TP_bindding {
	
	//calculate the soccz4 quantites use the real physics quantities given in TP lib
	inline void SOCCZ4Cal(double* __restrict__ Q){
		// Input: the physical metric gamma_ij (Q[0..5]) and the physical
		// extrinsic curvature K_ij (Q[6..11]), both as upper triangles.
		// Output: the conformal metric, Aex, Kex and phi.
		const double det = Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] -Q[2]*Q[2]*Q[3];
		const double invdet = 1./det;

		// Inverse of the physical metric (symmetric, i.e. upper triangle only)
		const double g_contr_11 = ( Q[3]*Q[5]-Q[4]*Q[4])*invdet;
		const double g_contr_12 = -( Q[1]*Q[5]-Q[2]*Q[4])*invdet;
		const double g_contr_13 = -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet;
		const double g_contr_22 = ( Q[0]*Q[5]-Q[2]*Q[2])*invdet;
		const double g_contr_23 = -( Q[0]*Q[4]-Q[2]*Q[1])*invdet;
		const double g_contr_33 = ( Q[0]*Q[3]-Q[1]*Q[1])*invdet;

		const double phi=std::pow(det,-1./6.);
		const double phi2=phi*phi;

		const double* Kdown = Q + 6;
		const double Kex = g_contr_11 * Kdown[0]
		                 + g_contr_22 * Kdown[3]
		                 + g_contr_33 * Kdown[5]
		                 + 2.0 * ( g_contr_12 * Kdown[1]
		                         + g_contr_13 * Kdown[2]
		                         + g_contr_23 * Kdown[4] );

		//now adjust the solution: Aex_ij = phi^2 (K_ij - 1/3 Kex gamma_ij)
		for (int i=0;i<6;i++) {
			Q[i+6]=	phi2*(Q[i+6]-1./3. * Kex * Q[i]); //\tilde{A}_ij
			Q[i]=phi2*Q[i]; //\tilde{\gamma}_ij
		}
		
		//Q[23]=Kex; Q[24]=std::log(phi); Q[16]=std::log(Q[16]);
		const double np = 1.0;
		Q[23]=Kex; Q[24]=std::pow(phi,np); //Q[16]=phi;
	}

	//use this to calculate the gradient
	inline void GradientCal(const double* X, double* __restrict__ Q, double* LgradQ, int nVars, TP::TwoPunctures* tp, bool low_res=false)
        {
		constexpr double epsilon = 1e-4;
		double Qp1[nVars],Qm1[nVars],Qp2[nVars],Qm2[nVars];
		
		for (int d=0;d<3;d++)
                {
			double xp1[3]={X[0],X[1],X[2]}; double xp2[3]={X[0],X[1],X[2]};
			double xm1[3]={X[0],X[1],X[2]}; double xm2[3]={X[0],X[1],X[2]};
			xp1[d]+=epsilon; xp2[d]+=2*epsilon; xm1[d]-=epsilon; xm2[d]-=2*epsilon;
			
			tp->Interpolate(xp1,Qp1, low_res); SOCCZ4Cal(Qp1);
			tp->Interpolate(xp2,Qp2, low_res); SOCCZ4Cal(Qp2);
			tp->Interpolate(xm1,Qm1, low_res); SOCCZ4Cal(Qm1);
			tp->Interpolate(xm2,Qm2, low_res); SOCCZ4Cal(Qm2);
			
			for (int i=0; i<nVars; i++) {
				LgradQ[d*nVars+i] = ( 8.0*Qp1[i] - 8.0*Qm1[i]  + Qm2[i]   - Qp2[i]  )/(12.0*epsilon);
			}
		}
	}
	
	//calculate the auxiliary variables using Q and its gradient
	inline void AuxiliaryCal(double* __restrict__ Q, double* LgradQ, int nVars){
		// LgradQ[d*nVars+i] is the derivative of entry i along axis d
		auto gradQ = [&](int d, int i) { return LgradQ[d*nVars+i]; };

		//here all quantites are coformal, i.e. with the tilde
		const double invdet = 1./(Q[0]*Q[3]*Q[5] - Q[0]*Q[4]*Q[4] - Q[1]*Q[1]*Q[5] + 2*Q[1]*Q[2]*Q[4] -Q[2]*Q[2]*Q[3]);
		const double g_contr_12 = -( Q[1]*Q[5]-Q[2]*Q[4])*invdet;
		const double g_contr_13 = -(-Q[1]*Q[4]+Q[2]*Q[3])*invdet;
		const double g_contr_23 = -( Q[0]*Q[4]-Q[2]*Q[1])*invdet;
		const double g_contr[3][3] = {{(Q[3]*Q[5]-Q[4]*Q[4])*invdet, g_contr_12, g_contr_13},
		                              {g_contr_12, (Q[0]*Q[5]-Q[2]*Q[2])*invdet, g_contr_23},
		                              {g_contr_13, g_contr_23, (Q[0]*Q[3]-Q[1]*Q[1])*invdet}};

		//A[i]=A_i=\partial_i \alpha
		for (int i=0;i<3;i++) Q[25+i]=gradQ(i,16);

		//B[i][j]=B_i^j=\partial_i \beta^j //the component order of this variable is different between fortran and C++!
		for (int i=0;i<3;i++)
		for (int j=0;j<3;j++) Q[28+i*3+j]=gradQ(i,17+j);

		//D[k][i][j]=D_kij= 0.5 * \partial_k \tilde{\gamma}_ij
		for (int i=0;i<3;i++)
		for (int j=0;j<6;j++) Q[37+i*6+j]=0.5*gradQ(i,j);

		//P[i]=P_i=\partial_i \phi
		for (int i=0;i<3;i++) Q[55+i]=gradQ(i,24);

		const double DD[3][3][3] = {
		  {{Q[37], Q[38], Q[39]}, {Q[38], Q[40], Q[41]}, {Q[39], Q[41], Q[42]}},
		  {{Q[43], Q[44], Q[45]}, {Q[44], Q[46], Q[47]}, {Q[45], Q[47], Q[48]}},
		  {{Q[49], Q[50], Q[51]}, {Q[50], Q[52], Q[53]}, {Q[51], Q[53], Q[54]}}};

		// Christoffel symbols, symmetric in i and j: Christoffel_tilde_down[l][i][j]
		// = D_ijl + D_jil - D_lij, raised to Christoffel_tilde[i][j][k]
		double Christoffel_tilde_down[3][3][3];
		double Christoffel_tilde[3][3][3];
		for (int i = 0; i < 3; i++)
		for (int j = i; j < 3; j++)
		{
		    for (int l = 0; l < 3; l++)
		    {
		        Christoffel_tilde_down[l][i][j] = DD[i][j][l] + DD[j][i][l] - DD[l][i][j];
		        Christoffel_tilde_down[l][j][i] = Christoffel_tilde_down[l][i][j];
		    }
		    for (int k = 0; k < 3; k++)
		    {
		        double temp = 0;
		        for (int l = 0; l < 3; l++) temp += g_contr[k][l] * Christoffel_tilde_down[l][i][j];
		        Christoffel_tilde[i][j][k] = temp;
		        Christoffel_tilde[j][i][k] = temp;
		    }
		}

		// Ghat^i = gamma^jl Christoffel^i_jl, a contraction of two symmetric index pairs
		for (int i = 0; i < 3; i++)
		{
		    Q[13+i] += g_contr[0][0] * Christoffel_tilde[0][0][i]
		             + g_contr[1][1] * Christoffel_tilde[1][1][i]
		             + g_contr[2][2] * Christoffel_tilde[2][2][i]
		             + 2.0 * ( g_contr[0][1] * Christoffel_tilde[0][1][i]
		                     + g_contr[0][2] * Christoffel_tilde[0][2][i]
		                     + g_contr[1][2] * Christoffel_tilde[1][2][i] );
		}
    }
}


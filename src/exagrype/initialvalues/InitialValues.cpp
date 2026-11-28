#include "exagrype/initialvalues/InitialValues.h"

#include "tarch/la/Vector.h"

#include "peano4/utils/Globals.h"

#include <stdio.h>
#include <string.h>


void exagrype::initialvalues::gaugeWaveEvolvedVariables(
  double * __restrict__ Q, // Q[25]
  const tarch::la::Vector<Dimensions,double>&  x,
  double t
) {
  constexpr int nVars = 25;
  constexpr double pi = M_PI;
  constexpr double peak_number = 2.0;
  constexpr double ICA = 0.1; ///< Amplitude of the wave 
  double HH     = 1.0 - ICA*sin( peak_number*pi*( x[0] - t));
  double dxH    = -peak_number*pi*ICA*cos( peak_number * pi*(x[0] - t));
  double phi    = pow(( 1.0 / HH),(1.0/6.0));
  double Kxx    = - 0.5*peak_number*pi*ICA*cos( peak_number * pi*(x[0] - t))/sqrt( 1.0 - ICA*sin( peak_number*pi*( x[0] - t))  );
  double Kex = Kxx/HH;
  tarch::memset(Q, .0, nVars*sizeof(double));
  Q[0]  = phi*phi*HH ;					//\tilde(\gamma)_xx
  Q[3]  = phi*phi  ;					//\tilde(\gamma)_yy
  Q[5]  = phi*phi                            ;		//\tilde(\gamma)_zz
  Q[6]  = phi*phi*(Kxx - 1.0/3.0*Kex*HH ) ;		//\tilde(A)_xx
  Q[9] =  phi*phi*(0.0 - 1.0/3.0*Kex*1.0) ;		//\tilde(A)_yy
  Q[11] = phi*phi*(0.0 - 1.0/3.0*Kex*1.0) ;		//\tilde(A)_zz
  Q[16] = sqrt(HH);				        //\alpha
  Q[13] = 2.0/(3.0*pow(HH,(5.0/3.0)))*dxH        ;	//\hat(\Gamma)^x
  Q[23] = Kex;					//K
  Q[24] = phi;					//\phi
}

void exagrype::initialvalues::gaugeWave(
  double * __restrict__ Q, // Q[58+0],
  const tarch::la::Vector<Dimensions,double>&  x,
  double t
) {
  gaugeWaveEvolvedVariables(Q, x, t);

  tarch::memset(Q+25, .0, (58-25)*sizeof(double));

  constexpr double pi = M_PI;
  constexpr double peak_number = 2.0;
  constexpr double ICA = 0.1; ///< Amplitude of the wave
  double HH     = 1.0 - ICA*sin( peak_number*pi*( x[0] - t));
  double dxH    = -peak_number*pi*ICA*cos( peak_number * pi*(x[0] - t));
  double dxphi  = - pow(HH,(-7.0/6.0))*dxH/6.0;
  double phi    = pow(( 1.0 / HH),(1.0/6.0));
  Q[25] = 1.0/(2.0)*dxH*pow(HH,(-1.0/2.0))       ;		//A_x
  Q[37] = pow(HH,(-1.0/3.0))*dxH/3.0         ;		//D_xxx
  Q[40] = phi*dxphi                     ;		//D_xyy
  Q[42] = phi*dxphi                    ;		//D_xzz
  Q[55] = dxphi;					//P_x
}

void exagrype::initialvalues::diagonal_gaugeWaveEvolvedVariables(
  double * __restrict__ Q, // Q[25]
  const tarch::la::Vector<Dimensions,double>&  x,
  double t
) {
  constexpr int nVars = 25;
  constexpr double pi = M_PI;
  constexpr double peak_number = 2.0;
  constexpr double ICA = 0.1; ///< Amplitude of the wave, B in the note 
  double phase  = peak_number*pi*( x[0] - x[1] - t);
  double H=1+2.0*ICA*sin(phase);
  tarch::memset(Q, .0, nVars*sizeof(double));
  Q[0]  = pow(H,(-1.0/3.0))*(1+ICA*sin(phase));				//\tilde(\gamma)_xx
  Q[1]  = -pow(H,(-1.0/3.0))*(ICA*sin(phase));				//\tilde(\gamma)_xy
  Q[3]  = pow(H,(-1.0/3.0))*(1+ICA*sin(phase));				//\tilde(\gamma)_yy
  Q[5]  = pow(H,(-1.0/3.0));						//\tilde(\gamma)_zz
  Q[6]  = pow(H,(-5.0/6.0))*(peak_number*pi*ICA/2.0)*cos(phase)*(1.0 - 2.0 * (1+ICA*sin(phase)) / (3.0*H) );	//\tilde(A)_xx
  Q[7]  = pow(H,(-5.0/6.0))*(peak_number*pi*ICA/2.0)*cos(phase)*(2.0 * ICA*sin(phase) / (3.0*H) - 1.0 ); 	//\tilde(A)_xy
  Q[9]  = pow(H,(-5.0/6.0))*(peak_number*pi*ICA/2.0)*cos(phase)*(1.0 - 2.0 * (1+ICA*sin(phase)) / (3.0*H) );	//\tilde(A)_yy
  Q[11] = pow(H,(-11.0/6.0))*(peak_number*pi*ICA/3.0)*cos(phase);						//\tilde(A)_zz
  Q[13] = pow(H,(-5.0/3.0))*(4.0*peak_number*pi*ICA/3.0)*cos(phase);	//\hat(\Gamma)^x
  Q[14] = -pow(H,(-5.0/3.0))*(4.0*peak_number*pi*ICA/3.0)*cos(phase);	//\hat(\Gamma)^y
  Q[16] = 0.5*log(H);							//ln(\alpha)
  Q[23] = pow(H,(-3.0/2.0))*(peak_number*pi*ICA)*cos(phase);		//K
  Q[24] = -(1.0/6.0)*log(H);						//ln(\phi)
}

void exagrype::initialvalues::diagonal_gaugeWave(
  double * __restrict__ Q, // Q[58+0],
  const tarch::la::Vector<Dimensions,double>&  x,
  double t
) {
  diagonal_gaugeWaveEvolvedVariables(Q, x, t);

  tarch::memset(Q+25, .0, (58-25)*sizeof(double));

  constexpr double pi = M_PI;
  constexpr double peak_number = 2.0;
  constexpr double ICA = 0.1; ///< Amplitude of the wave, B in the note
  double phase  = peak_number*pi*( x[0] - x[1] - t);
  double H=1+2.0*ICA*sin(phase);
  Q[25] = pow(H,-1.0)*(peak_number*pi*ICA)*cos(phase);			//A_x					
  Q[26] = -pow(H,-1.0)*(peak_number*pi*ICA)*cos(phase);			//A_y	
  Q[37] = pow(H,(-4.0/3.0))*(peak_number*pi*ICA/6.0)*cos(phase)*(1.0 + 4.0 * ICA*sin(phase) );		//D_xxx
  Q[38] = -pow(H,(-4.0/3.0))*(peak_number*pi*ICA/6.0)*cos(phase)*(3.0 + 4.0 * ICA*sin(phase) );		//D_xxy
  Q[40] = pow(H,(-4.0/3.0))*(peak_number*pi*ICA/6.0)*cos(phase)*(1.0 + 4.0 * ICA*sin(phase) );		//D_xyy
  Q[42] = -pow(H,(-4.0/3.0))*(peak_number*pi*ICA/3.0)*cos(phase);					//D_xzz
  Q[43] = -pow(H,(-4.0/3.0))*(peak_number*pi*ICA/6.0)*cos(phase)*(1.0 + 4.0 * ICA*sin(phase) );		//D_yxx
  Q[44] = pow(H,(-4.0/3.0))*(peak_number*pi*ICA/6.0)*cos(phase)*(3.0 + 4.0 * ICA*sin(phase) );		//D_yxy
  Q[43] = -pow(H,(-4.0/3.0))*(peak_number*pi*ICA/6.0)*cos(phase)*(1.0 + 4.0 * ICA*sin(phase) );		//D_yyy
  Q[48] = pow(H,(-4.0/3.0))*(peak_number*pi*ICA/3.0)*cos(phase);					//D_yzz
  Q[55] = -pow(H,-1.0)*(peak_number*pi*ICA/3.0)*cos(phase);		//P_x
  Q[56] = pow(H,-1.0)*(peak_number*pi*ICA/3.0)*cos(phase);		//P_x
}

void exagrype::initialvalues::linearWaveEvolvedVariables(
  double * __restrict__ Q, // Q[25]
  const tarch::la::Vector<Dimensions,double>&  X,
  double t
) {
  constexpr int nVars = 25;
  constexpr double pi = M_PI;
  constexpr double peak_number = 2.0;
  constexpr double ICA = 1e-4; ///< Amplitude of the wave, should be very small to keep linearized.
  double HH     =  ICA*sin( peak_number*pi*( X[0] - t));
  double dtHH   = -peak_number*pi*ICA*cos( peak_number * pi*(X[0] - t));
  tarch::memset(Q, .0, nVars*sizeof(double));
  Q[0]  = 1.0  ;		//\tilde(\gamma)_xx
  Q[3]  = 1+HH ;		//\tilde(\gamma)_yy
  Q[5]  = 1-HH ;		//\tilde(\gamma)_zz
  Q[6]  = 0.0	    ;		//\tilde(A)_xx
  Q[9]  = -0.5*dtHH ;		//\tilde(A)_yy
  Q[11] = 0.5*dtHH  ;		//\tilde(A)_zz
  Q[16] = log(1.0) ;		//ln(\alpha)
  Q[24] = log(1.0) ;		//ln(\phi)
}

void exagrype::initialvalues::linearWave(
  double * __restrict__ Q, // Q[58+0],
  const tarch::la::Vector<Dimensions,double>&  X,
  double t
) {
  linearWaveEvolvedVariables(Q, X, t);

  tarch::memset(Q+25, .0, (58-25)*sizeof(double));

  constexpr double pi = M_PI;
  constexpr double peak_number = 2.0;
  constexpr double ICA = 1e-4; ///< Amplitude of the wave, should be very small to keep linearized.
  double dxHH   =  peak_number*pi*ICA*cos( peak_number * pi*(X[0] - t));
  Q[37] = 0.0       ;		//D_xxx
  Q[40] = 0.5*dxHH  ;		//D_xyy
  Q[42] = -0.5*dxHH ;		//D_xzz
}

void exagrype::initialvalues::flatEvolvedVariables(
  double * __restrict__ Q, // Q[25]
  const tarch::la::Vector<Dimensions,double>&  x,
  double t
) {
  tarch::memset(Q, .0, 25*sizeof(double));
  Q[0]  = 1.0 ;          //\tilde(\gamma)_xx
  Q[3]  = 1.0  ;          //\tilde(\gamma)_yy
  Q[5]  = 1.0                            ;    //\tilde(\gamma)_zz
  Q[16] = 1.0;
  Q[24] = 1.0;          //\phi
}

void exagrype::initialvalues::flat(
  double * __restrict__ Q, // Q[58+0],
  const tarch::la::Vector<Dimensions,double>&  x,
  double t
) {
  flatEvolvedVariables(Q, x, t);

  tarch::memset(Q+25, .0, (58-25)*sizeof(double));
}

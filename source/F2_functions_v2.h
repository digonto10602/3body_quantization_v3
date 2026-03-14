#ifndef F2FUNCTIONSV2_H
#define F2FUNCTIONSV2_H


#include<bits/stdc++.h>
#include<cmath>
#include<Eigen/Dense>
#include "functions.h"
//#include "gsl/gsl_sf_dawson.h"
//#include<Eigen/Dense>

//#include "Faddeeva.cc"
#include "Faddeeva.hh"
#include "spherical_functions.h"



/* First we code all the function needed for F2 functions, we start with a single F in S-wave
as needed to check. We follow the paper = https://arxiv.org/pdf/2111.12734.pdf */
/* This is the v2 of this code base, here we add the spherical harmonics to F2 functions 
such that in future it can take in any partial wave (ell<3) can build the F2 function or 
matrix as needed for the three-body QC */

typedef std::complex<double> comp;

comp I0F(   comp En, 
            comp sigma_p,
            comp p,
            comp total_P, 
            double alpha,
            double mi,
            double mj,
            double mk, 
            double L )
{
    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi); 
    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;

    comp A = ((comp)4.0)*pi*gamma;
    comp B = -std::sqrt(pi/((comp)alpha))*((comp)1.0/(comp)2.0)*std::exp((comp)alpha*x*x);
    //comp C = 0.5*(pi*x)*ERFI_func(std::sqrt(alpha*x*x));

    double relerr = 0.0;
    comp term1 = ((comp)alpha)*x*x; 
    
    std::complex<double> term1comp(term1.real(), term1.imag()); 
    comp fadeeva_erfi = Faddeeva::erfi(std::sqrt(term1comp),relerr);

    comp C = ((comp)0.5)*(pi*x)*fadeeva_erfi;

    char debug = 'n';
    if(debug=='y')
    {
        std::cout<<std::setprecision(30);
        std::cout<<"constant = "<<A<<std::endl;
        std::cout<<"Fadeeva Erfi = "<<fadeeva_erfi<<std::endl; 
        std::cout<<"factor1 = "<<B<<std::endl; 
        std::cout<<"factor2 = "<<C<<std::endl; 
    }
    return A*(B + C);

}

comp I1F(   comp En, 
            comp sigma_p,
            comp p,
            comp total_P, 
            double alpha,
            double mi,
            double mj,
            double mk, 
            double L )
{
    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi); 
    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;

    comp A = ((comp)4.0)*pi*gamma;
    comp term = pi/(((comp) alpha)*((comp) alpha)*((comp) alpha)); 
    comp B = -std::sqrt(pi/(term)) * (((comp)1.0) + ((comp)2.0)*((comp)alpha)*x*x)/((comp)4.0) * std::exp(((comp)alpha)*x*x); 
    
    //comp C = 0.5*(pi*x)*ERFI_func(std::sqrt(alpha*x*x));

    comp term1 = ((comp)alpha)*x*x; 
    double relerr = 0.0;

    
    std::complex<double> term1comp(term1.real(), term1.imag()); 
    comp fadeeva_erfi = Faddeeva::erfi(std::sqrt(term1comp),relerr);

    comp C = ((comp)0.5)*(pi*x*x*x)*fadeeva_erfi;

    char debug = 'n';
    if(debug=='y')
    {
        std::cout<<std::setprecision(30);
        std::cout<<"constant = "<<A<<std::endl;
        std::cout<<"factor1 = "<<B<<std::endl; 
        std::cout<<"factor2 = "<<C<<std::endl; 
    }
    return A*(B + C);

}

comp I2F(   comp En, 
            comp sigma_p,
            comp p,
            comp total_P, 
            double alpha,
            double mi,
            double mj,
            double mk, 
            double L )
{
    comp pi = std::acos(-1.0);
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi); 
    comp gamma = (En - omega_func(p,mi))/std::sqrt(sigma_p);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;

    comp A = ((comp)4.0)*pi*gamma;
    comp B = -std::sqrt(pi/(((comp)alpha)*((comp)alpha)*((comp)alpha)*((comp)alpha)*((comp)alpha))) * (((comp)3.0) + ((comp)2.0)*((comp)alpha)*x*x + ((comp)4.0)*((comp)alpha)*((comp)alpha)*x*x*x*x)/((comp)8.0) * std::exp(((comp)alpha)*x*x); 
    
    //comp C = 0.5*(pi*x)*ERFI_func(std::sqrt(alpha*x*x));
    comp term1 = ((comp)alpha)*x*x;
    double relerr = 0.0;

    std::complex<double> term1comp(term1.real(), term1.imag()); 
    comp fadeeva_erfi = Faddeeva::erfi(std::sqrt(term1comp),relerr);

    comp C = ((comp)0.5)*(pi*x*x*x*x*x)*fadeeva_erfi;

    char debug = 'n';
    if(debug=='y')
    {
        std::cout<<std::setprecision(25);
        std::cout<<"constant = "<<A<<std::endl;
        std::cout<<"factor1 = "<<B<<std::endl; 
        std::cout<<"factor2 = "<<C<<std::endl; 
    }
    return A*(B + C);

}

comp I_int_ang_mom( 
                    comp En, 
                    comp sigma_p,
                    comp p,
                    comp total_P, 
                    int ell_f,
                    int proj_mf, 
                    int ell_i, 
                    int proj_mi, 
                    double alpha,
                    double mi,
                    double mj,
                    double mk, 
                    double L,
                    bool Q0norm //this is the Q0 normalization that removes q*^{ell} barrier factors 
                                //if true then it will use Q0 normalization, the terms will have explicit q*^{ell} dependence 
                )
{
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;
    
    comp zero_val = {0.0, 0.0};
    
    if(ell_f==ell_i)
    {
        if(proj_mf==proj_mi)
        {
            if(ell_i==0)
            {
                comp I0 = I0F(En, sigma_p, p, total_P, alpha, mi, mj, mk, L); 
                return I0; 
            }
            else if(ell_i==1)
            {
                comp I1 = I1F(En, sigma_p, p, total_P, alpha, mi, mj, mk, L); 
                if(Q0norm)
                {
                    I1 = I1*std::pow(twopibyL, 2.0*ell_i);
                }
                else
                {
                    I1 = I1/(std::pow(x,2.0*ell_i));
                }
                return I1;
            }
            else if(ell_i==2)
            {
                comp I2 = I2F(En, sigma_p, p, total_P, alpha, mi, mj, mk, L); 
                if(Q0norm)
                {
                    I2 = I2*(std::pow(twopibyL, 2.0*ell_i));
                }
                else
                {
                    I2 = I2/(std::pow(x,2.0*ell_i));
                }
                return I2;
            }
            else 
            {
                std::cerr << "Invalid ell=" << ell_i << " not added to the package yet, only upto ell=2 available!!!\n";
                return zero_val;
            }
        }
        else 
        {
            return zero_val; 
        }
    }
    else 
    {
        return zero_val;
    }
}

comp I_sum_ang_mom(   
                    comp En, 
                    comp sigma_p,
                    std::vector<comp> p,
                    std::vector<comp> total_P, 
                    int ell_f, 
                    int proj_mf, 
                    int ell_i, 
                    int proj_mi, 
                    double alpha,
                    double mi,
                    double mj,
                    double mk, 
                    double L,
                    int max_shell_num,
                    bool Q0norm
                )
{
    char debug = 'n';
    double tolerance = 1.0e-11;
    comp pi = std::acos(-1.0); 
    comp Lby2pi = ((comp) L)/((comp) 2.0*pi);
    comp twopibyL = ((comp) 2.0*pi)/((comp) L);

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp spec_p = std::sqrt(px*px + py*py + pz*pz);

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp gamma = (En - omega_func(spec_p,mi))/std::sqrt(sigma_p);
    //std::cout<<"gamma = "<<gamma<<std::endl;
    comp x = std::sqrt(q2psq_star(sigma_p,mj,mk))*Lby2pi;
    comp xii = ((comp)0.5)*(((comp)1.0) + ((comp)(mj*mj - mk*mk))/sigma_p); //this is another xi which is used inside the F sum function

    if(debug=='y')
    {
        std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;

        //std::cout<<"x = "<<x<<'\t'<<"sig_p = "<<sigma_p<<std::endl;
    }

    comp npPx = (px - Px)*Lby2pi; 
    comp npPy = (py - Py)*Lby2pi;
    comp npPz = (pz - Pz)*Lby2pi;

    comp npP = std::sqrt(npPx*npPx + npPy*npPy + npPz*npPz);

    int c1 = 0;
    int c2 = 0; //these two are for checking if p and P are zero or not

    if(abs(spec_p)<1.0e-10 || abs(spec_p)==0.0 ) c1 = 1;
    if(abs(total_P_val)<1.0e-10 || abs(total_P_val)==0.0) c2 = 1;

    comp xibygamma = xii/gamma; 
    
    //int max_shell_num = 50;
    int na_x_initial = -max_shell_num;
    int na_x_final = +max_shell_num;
    int na_y_initial = -max_shell_num;
    int na_y_final = +max_shell_num;
    int na_z_initial = -max_shell_num;
    int na_z_final = +max_shell_num;

    comp summ = {0.0,0.0};
    comp temp_summ = {0.0,0.0};
    for(int i=na_x_initial;i<na_x_final+1;++i)
    {
        for(int j=na_y_initial;j<na_y_final+1;++j)
        {
            for(int k=na_z_initial;k<na_z_final+1;++k)
            {
                comp na = (comp) std::sqrt(i*i + j*j + k*k);

                comp nax = (comp) i;
                comp nay = (comp) j;
                comp naz = (comp) k;

                comp nax_npPx = nax*npPx;
                comp nay_npPy = nay*npPy;
                comp naz_npPz = naz*npPz; 

                comp na_dot_npP = nax_npPx + nay_npPy + naz_npPz;
                comp npPsq = npP*npP; 

                comp prod1 = ( (na_dot_npP/npPsq)*(((comp)1.0)/gamma - ((comp)1.0)) + xibygamma );
                
                comp rx = 0.0;
                comp ry = 0.0;
                comp rz = 0.0;
                
                if(c1==1 && c2==1)
                {
                    rx = nax;
                    ry = nay;
                    rz = naz; 
                }
                else 
                {
                    if(abs(npPsq)==0)
                    {
                        rx = nax;
                        ry = nay;
                        rz = naz;
                    }
                    else 
                    {
                        rx = nax + npPx*prod1;
                        ry = nay + npPy*prod1;
                        rz = naz + npPz*prod1;
                    }
                    
                }

                comp r = std::sqrt(rx*rx + ry*ry + rz*rz);
                std::vector<comp> rvec(3); 
                rvec[0] = rx; 
                rvec[1] = ry; 
                rvec[2] = rz; 

                
                comp Ylm1 = spherical_harmonics(rvec, ell_f, proj_mf); 
                comp Ylm2 = spherical_harmonics(rvec, ell_i, proj_mi);
                comp term =  Ylm1 * std::exp(((comp)alpha)*(x*x - r*r))/(x*x - r*r) * Ylm2;
                comp prev_term = summ; 
                summ = summ + Ylm1 * std::exp(((comp)alpha)*(x*x - r*r))/(x*x - r*r) * Ylm2;

                if(debug=='y')
                {
                    if((int) nax.real()==0 && (int) nay.real()==2 && (int) naz.real()==2)
                    {
                    if(ell_f==1 && proj_mf==-1 && ell_i==1 && proj_mi==-1)
                    {
                        std::cout << "-------------------------" << std::endl;
                        std::cout << std::setprecision(30);
                        std::cout << "na_vec = [" << nax << "," 
                                                  << nay << "," 
                                                  << naz << "]" 
                                                  << std::endl; 
                        std::cout << "rvec = [" << rx << "," 
                                                << ry << "," 
                                                << rz << "]" 
                                                << std::endl; 
                        std::cout << "Ylm1 : " << Ylm1 << '\t'
                                  << "Ylm2 : " << Ylm2 << std::endl; 
                        std::cout << "pow term : " << (std::pow(twopibyL, (comp) ell_f +  (comp) ell_i)) << std::endl; 
                        std::cout << "UV+prop : " << std::exp((comp)alpha*(x*x - r*r))/(x*x - r*r) << std::endl;
                        std::cout << "summand term : " << term << std::endl; 
                        std::cout << "prev out : " << prev_term * (std::pow(twopibyL, ell_f + ell_i)) << std::endl; 
                        std::cout << "out : " << term*(std::pow(twopibyL, ell_f + ell_i)) << std::endl; 
                        comp term_some_A = term * (std::pow(twopibyL, ell_f + ell_i));
                        comp term_some_B = prev_term * (std::pow(twopibyL, ell_f + ell_i));
                        
                        std::cout << "prev sum term : " << term_some_B << std::endl;
                        std::cout << "curr sum term : " << term_some_A << std::endl; 
                        std::cout << "sum check : " <<  term_some_A.real() + term_some_B.real() << std::endl; 
                        std::cout << "sum term : " << summ*(std::pow(twopibyL, ell_f + ell_i)) <<std::endl;
                    }
                    
                     
                    //std::cout<<i<<'\t'<<j<<'\t'<<k<<'\t'<<x*x - r*r<<'\t'<<prod1<<'\t'<<summ<<std::endl;
                    if(!std::isnan(abs(prod1)))
                    {
                        //std::cout<<"npP = "<<npP<<'\t'
                        //     <<"prod1 = "<<prod1<<'\t'
                        //     <<"r = "<<r<<'\t'<<"rx = "<<rx<<'\t'<<"ry = "<<ry<<'\t'<<"rz = "<<rz<<std::endl;
                    }

                    std::cout << "-------------------------" << std::endl;
                    }
                }
                
            }
        }
    }

    if(Q0norm)
    {
        summ = summ*(std::pow(twopibyL, ell_f + ell_i));
    }
    else
    {
        summ = summ/(std::pow(x, ell_f + ell_i));
    }

    if(debug=='y')
    {
        std::cout << "out = " << summ << std::endl; 
    }

    return summ ;
}

//We write the F2(p, ell_f, m_f; k, ell_i, m_i) function here

comp F2_ang_mom(    
                    comp En, 
                    std::vector<comp> k, //we assume that k,p,P are a 3-vector
                    std::vector<comp> p,
                    std::vector<comp> total_P,
                    int ell_f, 
                    int proj_mf,
                    int ell_i, 
                    int proj_mi, 
                    double L,
                    double mi,
                    double mj, 
                    double mk, 
                    double alpha,
                    double epsilon_h,
                    int max_shell_num,
                    bool Q0norm    
                )
{
    char debug = 'n';
    comp kx = k[0];
    comp ky = k[1];
    comp kz = k[2];

    comp px = p[0];
    comp py = p[1];
    comp pz = p[2];

    comp Px = total_P[0];
    comp Py = total_P[1];
    comp Pz = total_P[2];

    comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
    comp spec_p = std::sqrt(px*px + py*py + pz*pz);
    comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);

    comp sigp = sigma_pvec_based(En,p,mi,total_P);//sigma(En, spec_p, mi, total_P_val);

    comp cutoff = cutoff_function_1(sigp, mj, mk, epsilon_h);

    comp omega_p = omega_func(spec_p,mi);

    if(debug=='y')
    {
        std::cout << "================================" << std::endl; 
        
        std::cout << " F2_ang_mom function : " << std::endl; 
        std::cout << "--------------------------------" << std::endl; 
        std::cout << "p1x = " << px << '\t'
                  << "p1y = " << py << '\t'
                  << "p1z = " << pz << std::endl; 
        std::cout << "ell_f = " << ell_f << '\t'
                  << "proj_mf = " << proj_mf << std::endl; 
        std::cout << "k1x = " << kx << '\t'
                  << "k1y = " << ky << '\t'
                  << "k1z = " << kz << std::endl;
        std::cout << "ell_i = " << ell_i << '\t'
                  << "proj_mi = " << proj_mi << std::endl; 
        std::cout << "spec_k = " << spec_k << '\t'
                  << "spec_p = " << spec_p << std::endl; 
        std::cout << "total_P = " << total_P_val << std::endl; 
        std::cout << "sig_p = " << sigp << '\t'
                  << "cutoff = " << cutoff << '\t' 
                  << "omega_p = " << omega_p << std::endl; 
    }

    //condition for the delta function
    int condition_delta = 0;

    if(px==kx && py==ky && pz==kz)
    {
        condition_delta = 0;
    }
    else 
    {
        condition_delta = 1;
    }

    if(debug=='y')
    {
        std::cout << "condition delta = " << condition_delta << std::endl; 
    }
    
    if(condition_delta==1) return 0.0;
    else 
    {
        comp pi = std::acos(-1.0);
        comp A = cutoff/(((comp)16.0)*pi*pi*((comp)L)*((comp)L)*((comp)L)*((comp)L)*omega_p*(En - omega_p));

        comp B = I_sum_ang_mom(En, sigp, p, total_P, ell_f, proj_mf, ell_i, proj_mi, alpha, mi, mj, mk, L, max_shell_num, Q0norm);

        comp C = I_int_ang_mom(En, sigp, spec_p, total_P_val, ell_f, proj_mf, ell_i, proj_mi, alpha, mi, mj, mk, L, Q0norm);
        //std::cout<<A<<'\t'<<B<<'\t'<<C<<std::endl; 

        if(debug=='y')
        {
            std::cout << "cutoff times constant = " << A << std::endl;  
            std::cout << "sum = " << B << std::endl; 
            std::cout << "analytical res = " << C << std::endl; 
            std::cout << std::endl; 
            std::cout << "================================" << std::endl; 
        
        }

        if(abs(A)==0)
        {
            return 0.0;
        }
        else
        return A*(B - C);
    }

}


//This function builds the F2_i matrix in the channel, momentum and angular momentum l,m space
void F2_i_ang_mom_mat(  
                        Eigen::MatrixXcd &F2,
                        comp En, 
                        std::vector<std::vector<comp> > &plm_config,
                        std::vector<std::vector<comp> > &klm_config,
                        std::vector<comp> total_P,
                        double mi,
                        double mj, 
                        double mk, 
                        double L, 
                        double alpha, 
                        double epsilon_h,
                        int max_shell_num,
                        bool Q0norm
                      )
{
    char debug = 'n';
    char special_check = 'n'; 

    if(debug=='y')
    {
        std::cout << "We will print out the components of F2 matrix" << std::endl;
        std::cout << "(F2_i_mat function from F2_functions.h)" << std::endl; 
    }
    for(int i=0; i<plm_config[0].size(); ++i)
    {
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
        for(int j=0; j<klm_config[0].size(); ++j)
        {
            comp px = plm_config[0][i];
            comp py = plm_config[1][i];
            comp pz = plm_config[2][i];
            int ell_f = static_cast<int> (plm_config[3][i].real()); 
            int proj_mf = static_cast<int> (plm_config[4][i].real()); 


            comp spec_p = std::sqrt(px*px + py*py + pz*pz);
            std::vector<comp> p(3);
            p[0] = px;
            p[1] = py;
            p[2] = pz; 

            comp kx = klm_config[0][j];
            comp ky = klm_config[1][j];
            comp kz = klm_config[2][j];
            int ell_i = static_cast<int> (klm_config[3][j].real()); 
            int proj_mi = static_cast<int> (klm_config[4][j].real()); 

            if(debug=='y' && special_check=='y')
            {
                std::cout << "special check" << std::endl; 
                std::cout << "ell_i = " << ell_i << '\t' 
                          << "klmconfig = " << klm_config[3][j] << std::endl;
            }

            comp spec_k = std::sqrt(kx*kx + ky*ky + kz*kz);
            std::vector<comp> k(3);
            k[0] = kx;
            k[1] = ky;
            k[2] = kz; 

            comp Px = total_P[0];
            comp Py = total_P[1];
            comp Pz = total_P[2];

            comp total_P_val = std::sqrt(Px*Px + Py*Py + Pz*Pz);
            
            comp F2_val = F2_ang_mom(En, k, p, total_P, ell_f, proj_mf, ell_i, proj_mi, L, mi, mj, mk, alpha, epsilon_h, max_shell_num, Q0norm);

            F2(i,j) = F2_val;

            if(debug=='y')
            {
                std::cout << "i = " << i << '\t' << "j = " << j << std::endl; 
                std::cout << "px = " << px << '\t'
                          << "py = " << py << '\t'
                          << "pz = " << pz << std::endl; 
                std::cout << "ell_f = " << ell_f << '\t'
                          << "proj_mf = " << proj_mf << std::endl; 
                std::cout << "kx = " << kx << '\t'
                          << "ky = " << ky << '\t'
                          << "kz = " << kz << std::endl;
                std::cout << "ell_i = " << ell_i << '\t'
                          << "proj_mi = " << proj_mi << std::endl;
                std::cout << "F2 val = " << F2_val << std::endl;
                std::cout << "-------------------------------------" << std::endl;
            }

            


        }
        if(debug=='y')
        {
            std::cout << "-------------------------------------" << std::endl; 
        }
    }

    if(debug=='y')
    {
        std::cout << "=========================================" << std::endl; 
    }
}

//This builds the F2 function as a matrix in channel space 
//We pass the waves for particular channel i, and it build the
//F1tilde and F2tilde matrices to form F2 matrix for the 
//(2+1) system, 
/*
    F2 = | F2_1 0 .. |
         | 0    F2_2 | 

*/
// The system has m1 and m2 particles such that threshold = 2m1 + m2
void F2_2plus1_mat(
                        Eigen::MatrixXcd &F2,
                        comp En, 
                        std::vector<std::vector<comp> > &plm_config,
                        std::vector<std::vector<comp> > &klm_config,
                        std::vector<comp> total_P,
                        double m1,
                        double m2, 
                        double L, 
                        double alpha, 
                        double epsilon_h,
                        int max_shell_num,
                        bool Q0norm

)
{
    char debug = 'n';
    double mi, mj, mk = 0.0; 
    if(debug=='y')
    {
        std::cout << "  F2 angular momentum matrix in channel space " << std::endl;
        std::cout << "(F2_ang_mom_mat function from F2_functions.h)" << std::endl; 
    }

    int size1 = plm_config[0].size(); 
    int size2 = klm_config[0].size(); 

    int total_size = size1 + size2; 

    Eigen::MatrixXcd F2_1(size1, size1); 
    Eigen::MatrixXcd F2_2(size2, size2); 

    //when i=1 
    mi = m1; 
    mj = m1; 
    mk = m2; 

    F2_i_ang_mom_mat(F2_1, En, plm_config, plm_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num, Q0norm); 

    //when i=2
    mi = m2; 
    mj = m1; 
    mk = m1; 

    F2_i_ang_mom_mat(F2_2, En, klm_config, klm_config, total_P, mi, mj, mk, L, alpha, epsilon_h, max_shell_num, Q0norm); 

    Eigen::MatrixXcd Filler0_12(size1,size2);
    Eigen::MatrixXcd Filler0_21(size2,size1);
    Eigen::MatrixXcd Filler0_22(size2,size2);

    Filler0_12 = Eigen::MatrixXcd::Zero(size1,size2);
    Filler0_21 = Eigen::MatrixXcd::Zero(size2,size1);
    Filler0_22 = Eigen::MatrixXcd::Zero(size2,size2);

    Eigen::MatrixXcd F2_mat(size1 + size2,size1 + size2);
    F2_mat <<   F2_1,   Filler0_12,
                Filler0_21, F2_2; 

    F2 = F2_mat; 

}


            





#endif
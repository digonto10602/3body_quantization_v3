#ifndef QCFUNCTIONS_V2_H
#define QCFUNCTIONS_V2_H
#include "functions.h"
#include "F2_functions_v2.h"
#include "K2_functions_v2.h"
#include "G_functions_v2.h"


typedef std::complex<double> comp;

//Basic Solvers that we will need in this program
void LinearSolver_3(	Eigen::MatrixXcd &A,
					Eigen::MatrixXcd &X,
					Eigen::MatrixXcd &B,
					double &relerr 			)
{
	X = A.partialPivLu().solve(B);

	relerr = (A*X - B).norm()/B.norm();
}

void LinearSolver_4(	Eigen::MatrixXcd &A,
					Eigen::MatrixXcd &X,
					Eigen::MatrixXcd &B,
					double &relerr 			)
{
	X = A.colPivHouseholderQr().solve(B);

	relerr = (A*X - B).norm()/B.norm();
}


void test_F3iso_ND_2plus1_mat_with_normalization_single_En(
							Eigen::MatrixXcd &F3mat,
							comp &F3iso,
                            Eigen::VectorXcd &state_vec, 
                            Eigen::MatrixXcd &F2mat,
                            Eigen::MatrixXcd &K2imat,
                            Eigen::MatrixXcd &Gmat, 
                            Eigen::MatrixXcd &Hmatinv, 
                            comp En, 
                            std::vector< std::vector<comp> > plm_config,
                            std::vector< std::vector<comp> > klm_config, 
                            std::vector<comp> total_P, 
                            double eta_i_1,
                            double eta_i_2, 
                            std::vector<std::vector<comp> > scatter_params_1,
                            std::vector<std::vector<comp> > scatter_params_2,
                            double m1,
                            double m2,  
                            double alpha, 
                            double epsilon_h, 
                            double L, 
                            int max_shell_num,
							bool Q0norm  

)
{
	int size1 = plm_config[0].size();
	int size2 = klm_config[0].size(); 

	F2_2plus1_mat( F2mat, En, plm_config, klm_config, total_P, m1, m2, L, alpha, epsilon_h, max_shell_num, Q0norm);

	K2inv_EREord2_2plus1_mat(K2imat, eta_i_1, eta_i_2, scatter_params_1, scatter_params_2, En, plm_config, klm_config, total_P, m1, m2, epsilon_h, L);

	G_2plus1_mat(Gmat, En, plm_config, klm_config, total_P, m1, m2, L, alpha, epsilon_h, max_shell_num, Q0norm);
            
	Eigen::MatrixXcd Hmat = K2imat + F2mat + Gmat; 

	int eigvec_counter = 0; 
    Eigen::VectorXcd EigVec(size1 + size2); 
    for(int i=0;i<size1;++i)
    {
        EigVec(i) = 1.0;
        eigvec_counter += 1; 
    }
    for(int i=eigvec_counter ;i<(size1+size2); ++i)
    {
        EigVec(i) = 1.0/std::sqrt(2.0); 
    }

	Eigen::MatrixXcd temp_identity_mat(size1 + size2,size1 + size2);
    temp_identity_mat.setIdentity();


	Eigen::MatrixXcd H_mat_inv(size1 + size2,size1 + size2);
    double relerror = 0.0;
	LinearSolver_4(Hmat, H_mat_inv, temp_identity_mat, relerror);

	Eigen::MatrixXcd F3_mat = (F2mat/3.0 - F2mat*H_mat_inv*F2mat);

	F3mat = F3_mat; 
	F3iso = EigVec.transpose()*F3mat*EigVec; 


	//std::cout<<"matrix size = "<<size1+size2 << "x "<<size1+size2<< std::endl;
                        


}

#endif
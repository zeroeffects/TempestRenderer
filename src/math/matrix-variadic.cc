/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2016 Zdravko Velinov
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in
*   all copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*   THE SOFTWARE.
*/

#include "tempest/math/matrix-variadic.hh"
#include "tempest/utils/assert.hh"
#include "eigen-eigen/Eigen/Core"
#include "eigen-eigen/Eigen/LU"
#include "eigen-eigen/Eigen/Cholesky"
#include "eigen-eigen/Eigen/QR"

namespace Tempest
{
template<class TFloat>
using VectorType = Eigen::Matrix<TFloat, Eigen::Dynamic, 1>;

template<class TFloat>
using MatrixType = Eigen::Matrix<TFloat, Eigen::Dynamic, Eigen::Dynamic>;

template<class TFloat>
using ArrayType = Eigen::Array<TFloat, Eigen::Dynamic, 1>;

template<class TFloat>
void VectorNegate(const TFloat* in_vec, uint32_t vec_size,  TFloat** out_vec)
{
	Eigen::Map<VectorType<TFloat>> e_in_vec(const_cast<TFloat*>(in_vec), vec_size);
	
	Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, vec_size);

	e_out_vec = -e_in_vec;
}

template<class TFloat>
void VectorTransposeMatrixTransform(const TFloat* in_vec, uint32_t vec_size, const TFloat* in_matrix, uint32_t row_count, uint32_t col_count, TFloat** out_vec)
{
	Eigen::Map<VectorType<TFloat>> e_in_vec(const_cast<TFloat*>(in_vec), vec_size);

	Eigen::Map<MatrixType<TFloat>> e_in_mat(const_cast<TFloat*>(in_matrix), row_count, col_count);

	Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, vec_size);

	e_out_vec = e_in_vec.transpose() * e_in_mat;
}

template<class TFloat>
void VectorOuterProduct(const TFloat* in_lhs_vec, uint32_t lhs_vec_size, const TFloat* in_rhs_vec, uint32_t rhs_vec_size, TFloat** out_matrix)
{
	Eigen::Map<VectorType<TFloat>> e_in_lhs_vec(const_cast<TFloat*>(in_lhs_vec), lhs_vec_size);
	Eigen::Map<VectorType<TFloat>> e_in_rhs_vec(const_cast<TFloat*>(in_rhs_vec), rhs_vec_size);

	Eigen::Map<MatrixType<TFloat>> e_out_mat(*out_matrix, lhs_vec_size, rhs_vec_size);

	e_out_mat = e_in_lhs_vec * e_in_rhs_vec.transpose();
}

template<class TFloat>
void MatrixMultiplyAdd(TFloat coefficient, const TFloat* lhs_mat, uint32_t lhs_row_count, uint32_t lhs_col_count,
					   const TFloat* rhs_mat, uint32_t rhs_row_count, uint32_t rhs_col_count, TFloat** out_matrix)
{
	Eigen::Map<MatrixType<TFloat>> e_lhs_mat(const_cast<TFloat*>(lhs_mat), lhs_row_count, lhs_col_count);
	Eigen::Map<MatrixType<TFloat>> e_rhs_mat(const_cast<TFloat*>(rhs_mat), rhs_row_count, rhs_col_count);

	Eigen::Map<MatrixType<TFloat>> e_out_mat(*out_matrix, lhs_row_count, lhs_col_count);

	e_out_mat = coefficient*e_lhs_mat + e_rhs_mat;
}

template<class TFloat>
bool MatrixCholeskyDecomposition(const TFloat* in_matrix, uint32_t row_col_count, TFloat** out_matrix)
{
	Eigen::Map<MatrixType<TFloat>> e_in_matrix(const_cast<TFloat*>(in_matrix), row_col_count, row_col_count);

	Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, row_col_count, row_col_count);

	auto llt = e_in_matrix.llt();

	if(llt.info() != Eigen::Success)
		return false;

	e_out_matrix = llt.matrixLLT();
	return true;
}

template<class TFloat>
void MatrixTransposeLinearSolve(const TFloat* in_lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_col_count,
                                const TFloat* in_rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_col_count, TFloat** out_matrix)
{
	Eigen::Map<MatrixType<TFloat>> e_in_lhs_matrix(const_cast<TFloat*>(in_lhs_matrix), lhs_row_count, lhs_col_count);
	Eigen::Map<MatrixType<TFloat>> e_in_rhs_matrix(const_cast<TFloat*>(in_rhs_matrix), rhs_row_count, rhs_col_count);

    auto qr_decompose = e_in_lhs_matrix.transpose().colPivHouseholderQr();
    Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, lhs_row_count, rhs_row_count);

	e_out_matrix = qr_decompose.solve(e_in_rhs_matrix.transpose());
}

template<class TFloat>
void MatrixTriangularSolve(const TFloat* in_matrix, uint32_t row_col_count, const TFloat* vec, uint32_t row_count, TFloat** out_vec)
{
	Eigen::Map<MatrixType<TFloat>> e_in_matrix(const_cast<TFloat*>(in_matrix), row_col_count, row_col_count);

	Eigen::Map<VectorType<TFloat>> e_vec(const_cast<TFloat*>(vec), row_count);

	Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, row_count);

	e_out_vec = e_in_matrix.template triangularView<Eigen::Lower>().solve(e_vec);
}

template<class TFloat>
void MatrixTriangularSolve(const TFloat* in_lhs_matrix, uint32_t row_col_count, const TFloat* in_rhs_matrix, uint32_t row_count, uint32_t col_count, TFloat** out_matrix)
{
    Eigen::Map<MatrixType<TFloat>> e_in_lhs_matrix(const_cast<TFloat*>(in_lhs_matrix), row_col_count, row_col_count);
    Eigen::Map<MatrixType<TFloat>> e_in_rhs_matrix(const_cast<TFloat*>(in_rhs_matrix), row_count, col_count);

    Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, row_count, col_count);

    e_out_matrix = e_in_lhs_matrix.template triangularView<Eigen::Lower>().solve(e_in_rhs_matrix);
}

template<class TFloat>
void MatrixTriangularTransposeSolve(const TFloat* in_matrix, uint32_t row_col_count, const TFloat* vec, uint32_t row_count, TFloat** out_vec)
{
	Eigen::Map<MatrixType<TFloat>> e_in_matrix(const_cast<TFloat*>(in_matrix), row_col_count, row_col_count);

	Eigen::Map<VectorType<TFloat>> e_vec(const_cast<TFloat*>(vec), row_count);

	Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, row_count);

	e_out_vec = e_in_matrix.template triangularView<Eigen::Lower>().transpose().solve(e_vec);
}

template<class TFloat>
void MatrixTransformCovarianceDiagonal(const TFloat* transform_matrix, uint32_t trans_row_count, uint32_t trans_column_count,
									   const TFloat* covariance_matrix, uint32_t cov_row_column_count, TFloat** out_matrix)
{
	Eigen::Map<MatrixType<TFloat>> e_trans_matrix(const_cast<TFloat*>(transform_matrix), trans_row_count, trans_column_count);

	Eigen::Map<VectorType<TFloat>> diag(const_cast<TFloat*>(covariance_matrix), cov_row_column_count);

	Eigen::DiagonalMatrix<TFloat, Eigen::Dynamic> e_covariance_matrix;
	e_covariance_matrix.diagonal() = diag; 

	MatrixType<TFloat> intermediate = e_trans_matrix*e_covariance_matrix;

	Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, trans_row_count, trans_row_count);
    e_out_matrix = intermediate*e_trans_matrix.transpose();
}

template<class TFloat>
void MatrixSquare(const TFloat* in_matrix, uint32_t row_count, uint32_t column_count, TFloat** out_matrix)
{
    Eigen::Map<MatrixType<TFloat>> e_in_matrix(const_cast<TFloat*>(in_matrix), row_count, column_count);

    auto actual_rows = e_in_matrix.rows();
    auto actual_cols = e_in_matrix.cols();
    TGE_ASSERT(row_count == e_in_matrix.rows() && column_count == e_in_matrix.cols(), "Invalid matrix");

    Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, row_count, row_count);
    e_out_matrix = e_in_matrix*e_in_matrix.transpose();
}

template<class TFloat>
void MatrixInverse(const TFloat* in_matrix, uint32_t row_col_count, TFloat** out_matrix)
{
	Eigen::Map<MatrixType<TFloat>> e_in_matrix(const_cast<TFloat*>(in_matrix), row_col_count, row_col_count);
    
    Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, row_col_count, row_col_count);
    e_out_matrix = e_in_matrix.inverse();
}

template<class TFloat>
void MatrixIdentity(uint32_t row_count, uint32_t col_count, TFloat** out_matrix)
{
	Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, row_count, col_count);
	e_out_matrix = MatrixType<TFloat>::Identity(row_count, col_count);
}

template<class TFloat>
void MatrixZeros(uint32_t row_count, uint32_t col_count, TFloat** out_matrix)
{
    Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, row_count, col_count);
    e_out_matrix = MatrixType<TFloat>::Zeros(row_count, col_count);
}


template<class TFloat>
void MatrixOnes(uint32_t row_count, uint32_t col_count, TFloat** out_matrix)
{
    Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, row_count, col_count);
    e_out_matrix = MatrixType<TFloat>::Ones(row_count, col_count);
}


template<class TFloat>
void MatrixMultiply(const TFloat* lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_column_count,
                    const TFloat* rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_column_count, TFloat** out_matrix)
{
    TGE_ASSERT(lhs_column_count == rhs_row_count, "Invalid matrices");

    Eigen::Map<MatrixType<TFloat>> e_lhs_matrix(const_cast<TFloat*>(lhs_matrix), lhs_row_count, lhs_column_count);
    Eigen::Map<MatrixType<TFloat>> e_rhs_matrix(const_cast<TFloat*>(rhs_matrix), rhs_row_count, rhs_column_count);

    Eigen::Map<MatrixType<TFloat>> e_out_matrix(*out_matrix, lhs_row_count, rhs_column_count);
    e_out_matrix = e_lhs_matrix*e_rhs_matrix;
}

template<class TFloat>
void MatrixTransformVector(const TFloat* in_matrix, uint32_t row_count, uint32_t column_count, const TFloat* vec, uint32_t vec_size, TFloat** out_vec)
{
	Eigen::Map<MatrixType<TFloat>> e_in_matrix(const_cast<TFloat*>(in_matrix), row_count, column_count);
    Eigen::Map<VectorType<TFloat>> e_vec(const_cast<TFloat*>(vec), vec_size);

    Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, row_count);
    e_out_vec = e_in_matrix*e_vec;
}

template<class TFloat>
TFloat VectorLength(const TFloat* in_vec, uint32_t vec_size)
{
    Eigen::Map<VectorType<TFloat>> e_in_vec(const_cast<TFloat*>(in_vec), vec_size);
    return e_in_vec.norm();
}

template<class TFloat>
TFloat VectorLengthSquared(const TFloat* in_vec, uint32_t vec_size)
{
    Eigen::Map<VectorType<TFloat>> e_in_vec(const_cast<TFloat*>(in_vec), vec_size);
    return e_in_vec.squaredNorm();
}

template<class TFloat>
void VectorSubtract(const TFloat* in_lhs, uint32_t lhs_size, const TFloat* in_rhs, uint32_t rhs_size, TFloat** out_vec)
{
    Eigen::Map<VectorType<TFloat>> e_in_lhs(const_cast<TFloat*>(in_lhs), lhs_size);
    Eigen::Map<VectorType<TFloat>> e_in_rhs(const_cast<TFloat*>(in_rhs), rhs_size);

    Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, lhs_size);
    e_out_vec = e_in_lhs - e_in_rhs;
}

template<class TFloat>
void VectorAdd(const TFloat* in_lhs, uint32_t lhs_size, const TFloat* in_rhs, uint32_t rhs_size, TFloat** out_vec)
{
    Eigen::Map<VectorType<TFloat>> e_in_lhs(const_cast<TFloat*>(in_lhs), lhs_size);
    Eigen::Map<VectorType<TFloat>> e_in_rhs(const_cast<TFloat*>(in_rhs), rhs_size);

    Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, lhs_size);
    e_out_vec = e_in_lhs + e_in_rhs;
}

template<class TFloat>
void VectorAdd(const TFloat* in_lhs, uint32_t lhs_size, TFloat scalar, TFloat** out_vec)
{
    Eigen::Map<ArrayType<TFloat>> e_in_lhs(const_cast<TFloat*>(in_lhs), lhs_size);
    
    Eigen::Map<ArrayType<TFloat>> e_out_vec(*out_vec, lhs_size);
    e_out_vec = e_in_lhs + scalar;
}

template<class TFloat>
TFloat VectorDot(const TFloat* in_lhs, uint32_t lhs_size, const TFloat* in_rhs, uint32_t rhs_size)
{
    Eigen::Map<VectorType<TFloat>> e_in_lhs(const_cast<TFloat*>(in_lhs), lhs_size);
    Eigen::Map<VectorType<TFloat>> e_in_rhs(const_cast<TFloat*>(in_rhs), rhs_size);

    return e_in_lhs.dot(e_in_rhs);
}

template<class TFloat>
void VectorMultiply(const TFloat* in_lhs, uint32_t vec_size, TFloat scalar, TFloat** out_vec)
{
    Eigen::Map<VectorType<TFloat>> e_in_lhs(const_cast<TFloat*>(in_lhs), vec_size);
    Eigen::Map<VectorType<TFloat>> e_out_vec(*out_vec, vec_size);
    e_out_vec = e_in_lhs*scalar;
}

template void VectorNegate(const float* in_vec, uint32_t vec_size, float** out_vec);
template void VectorTransposeMatrixTransform(const float* in_vec, uint32_t vec_size, const float* in_matrix, uint32_t row_count, uint32_t col_count, float** out_vec);
template void VectorOuterProduct(const float* in_lhs_vec, uint32_t lhs_vec_size, const float* in_rhs_vec, uint32_t rhs_vec_size, float** out_matrix);
template void MatrixIdentity(uint32_t row_count, uint32_t col_count, float** out_mat);
template void MatrixMultiplyAdd(float coefficient, const float* lhs_mat, uint32_t lhs_row_count, uint32_t lhs_col_count,
							    const float* rhs_mat, uint32_t rhs_row_count, uint32_t rhs_col_count, float** out_matrix);
template bool MatrixCholeskyDecomposition(const float* in_matrix, uint32_t row_col_count, float** out_matrix);
template void MatrixTransposeLinearSolve(const float* in_lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_col_count, const float* in_rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_col_count, float** out_matrix);
template void MatrixTriangularSolve(const float* in_matrix, uint32_t row_col_count, const float* vec, uint32_t row_count, float** out_vec);
template void MatrixTriangularSolve(const float* in_lhs_matrix, uint32_t row_col_count, const float* in_rhs_matrix, uint32_t row_count, uint32_t col_count, float** out_matrix);

template void MatrixTriangularTransposeSolve(const float* in_matrix, uint32_t row_col_count, const float* vec, uint32_t row_count, float** out_vec);
template void MatrixTransformCovarianceDiagonal(const float* transform_matrix, uint32_t trans_row_count, uint32_t trans_column_count,
											    const float* covariance_matrix, uint32_t cov_row_column_count, float** out_matrix);
template void MatrixSquare(const float* in_matrix, uint32_t row_count, uint32_t column_count, float** out_matrix);
template void MatrixInverse(const float* in_matrix, uint32_t row_col_count, float** out_matrix);
template void MatrixMultiply(const float* lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_column_count,
				             const float* rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_column_count, float** out_matrix);
template void MatrixTransformVector(const float* in_matrix, uint32_t row_count, uint32_t column_count, const float* vec, uint32_t vec_size, float** out_vec);
template float VectorLength(const float* in_vec, uint32_t vec_size);
template float VectorLengthSquared(const float* in_vec, uint32_t vec_size);
template void VectorSubtract(const float* in_lhs, uint32_t lhs_size, const float* in_rhs, uint32_t rhs_size, float** out_vec);
template void VectorAdd(const float* in_lhs, uint32_t lhs_size, const float* in_rhs, uint32_t rhs_size, float** out_vec);
template void VectorAdd(const float* in_lhs, uint32_t lhs_size, float scalar, float** out_vec);
template float VectorDot(const float* in_lhs, uint32_t lhs_size, const float* in_rhs, uint32_t rhs_size);
template void VectorMultiply(const float* in_lhs, uint32_t vec_size, float scalar, float** out_vec);

template void VectorNegate(const double* in_vec, uint32_t vec_size, double** out_vec);
template void VectorTransposeMatrixTransform(const double* in_vec, uint32_t vec_size, const double* in_matrix, uint32_t row_count, uint32_t col_count, double** out_vec);
template void VectorOuterProduct(const double* in_lhs_vec, uint32_t lhs_vec_size, const double* in_rhs_vec, uint32_t rhs_vec_size, double** out_matrix);
template void MatrixIdentity(uint32_t row_count, uint32_t col_count, double** out_mat);
template void MatrixMultiplyAdd(double coefficient, const double* lhs_mat, uint32_t lhs_row_count, uint32_t lhs_col_count,
							    const double* rhs_mat, uint32_t rhs_row_count, uint32_t rhs_col_count, double** out_matrix);
template bool MatrixCholeskyDecomposition(const double* in_matrix, uint32_t row_col_count, double** out_matrix);
template void MatrixTransposeLinearSolve(const double* in_lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_col_count, const double* in_rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_col_count, double** out_matrix);
template void MatrixTriangularSolve(const double* in_matrix, uint32_t row_col_count, const double* vec, uint32_t row_count, double** out_vec);
template void MatrixTriangularSolve(const double* in_lhs_matrix, uint32_t row_col_count, const double* in_rhs_matrix, uint32_t row_count, uint32_t col_count, double** out_matrix);
template void MatrixTriangularTransposeSolve(const double* in_matrix, uint32_t row_col_count, const double* vec, uint32_t row_count, double** out_vec);
template void MatrixTransformCovarianceDiagonal(const double* transform_matrix, uint32_t trans_row_count, uint32_t trans_column_count,
											    const double* covariance_matrix, uint32_t cov_row_column_count, double** out_matrix);
template void MatrixSquare(const double* in_matrix, uint32_t row_count, uint32_t column_count, double** out_matrix);
template void MatrixInverse(const double* in_matrix, uint32_t row_col_count, double** out_matrix);
template void MatrixMultiply(const double* lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_column_count,
				             const double* rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_column_count, double** out_matrix);
template void MatrixTransformVector(const double* in_matrix, uint32_t row_count, uint32_t column_count, const double* vec, uint32_t vec_size, double** out_vec);
template double VectorLength(const double* in_vec, uint32_t vec_size);
template double VectorLengthSquared(const double* in_vec, uint32_t vec_size);
template void VectorSubtract(const double* in_lhs, uint32_t lhs_size, const double* in_rhs, uint32_t rhs_size, double** out_vec);
template void VectorAdd(const double* in_lhs, uint32_t lhs_size, const double* in_rhs, uint32_t rhs_size, double** out_vec);
template void VectorAdd(const double* in_lhs, uint32_t lhs_size, double scalar, double** out_vec);
template double VectorDot(const double* in_lhs, uint32_t lhs_size, const double* in_rhs, uint32_t rhs_size);
template void VectorMultiply(const double* in_lhs, uint32_t vec_size, double scalar, double** out_vec);
}

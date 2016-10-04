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

#ifndef _TEMPEST_MATRIX_VARIADIC_HH_
#define _TEMPEST_MATRIX_VARIADIC_HH_

#include <cstdint>

namespace Tempest
{
// NOTE: Everything is COLUMN-MAJOR, so column then rows - the opposite of how arrays are expressed
template<class TFloat>
void VectorNegate(const TFloat* in_vec, uint32_t vec_size, TFloat** out_vec);

template<class TFloat>
void VectorTransposeMatrixTransform(const TFloat* in_vec, uint32_t vec_size, const TFloat* in_matrix, uint32_t row_count, uint32_t col_count, TFloat** out_vec);

template<class TFloat>
void VectorOuterProduct(const TFloat* in_lhs_vec, uint32_t lhs_vec_size, const TFloat* in_rhs_vec, uint32_t rhs_vec_size, TFloat** out_matrix);

template<class TFloat>
void MatrixIdentity(uint32_t row_count, uint32_t col_count, TFloat** out_mat);

template<class TFloat>
void MatrixZeros(uint32_t row_count, uint32_t col_count, TFloat** out_mat);

template<class TFloat>
void MatrixOnes(uint32_t row_count, uint32_t col_count, TFloat** out_mat);

template<class TFloat>
void MatrixMultiplyAdd(TFloat coefficient, const TFloat* lhs_mat, uint32_t lhs_row_count, uint32_t lhs_col_count,
					   const TFloat* rhs_mat, uint32_t rhs_row_count, uint32_t rhs_col_count, TFloat** out_matrix);

template<class TFloat>
bool MatrixCholeskyDecomposition(const TFloat* in_matrix, uint32_t row_col_count, TFloat** out_matrix);

template<class TFloat>
void MatrixTransposeLinearSolve(const TFloat* in_lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_col_count,
                                const TFloat* in_rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_col_count, TFloat** out_matrix);

template<class TFloat>
void MatrixTriangularSolve(const TFloat* in_matrix, uint32_t row_col_count, const TFloat* vec, uint32_t row_count, TFloat** out_vec);

template<class TFloat>
void MatrixTriangularSolve(const TFloat* in_matrix, uint32_t row_col_count, const TFloat* vec, uint32_t row_count, uint32_t col_count, TFloat** out_vec);

template<class TFloat>
void MatrixTriangularTransposeSolve(const TFloat* in_lhs_matrix, uint32_t row_col_count, const TFloat* in_rhs_matrix, uint32_t row_count, TFloat** out_vec);

template<class TFloat>
void MatrixTransformCovarianceDiagonal(const TFloat* transform_matrix, uint32_t trans_row_count, uint32_t trans_column_count,
									   const TFloat* covariance_matrix, uint32_t cov_row_column_count, TFloat** out_matrix);

template<class TFloat>
void MatrixSquare(const TFloat* in_matrix, uint32_t row_count, uint32_t column_count, TFloat** out_matrix);

template<class TFloat>
void MatrixInverse(const TFloat* in_matrix, uint32_t row_col_count, TFloat** out_matrix);

template<class TFloat>
void MatrixMultiply(const TFloat* lhs_matrix, uint32_t lhs_row_count, uint32_t lhs_column_count,
                    const TFloat* rhs_matrix, uint32_t rhs_row_count, uint32_t rhs_column_count, TFloat** out_matrix);

template<class TFloat>
void MatrixTransformVector(const TFloat* in_matrix, uint32_t row_count, uint32_t column_count, const TFloat* vec, uint32_t vec_size, TFloat** out_vec);

template<class TFloat>
TFloat VectorLength(const TFloat* in_vec, uint32_t vec_size);

template<class TFloat>
TFloat VectorLengthSquared(const TFloat* in_vec, uint32_t vec_size);

template<class TFloat>
void VectorSubtract(const TFloat* in_lhs, uint32_t lhs_size, const TFloat* in_rhs, uint32_t rhs_size, TFloat** out_vec);

template<class TFloat>
void VectorAdd(const TFloat* in_lhs, uint32_t lhs_size, const TFloat* in_rhs, uint32_t rhs_size, TFloat** out_vec);

template<class TFloat>
void VectorAdd(const TFloat* in_lhs, uint32_t lhs_size, TFloat scalar, TFloat** out_vec);

template<class TFloat>
TFloat VectorDot(const TFloat* in_lhs, uint32_t lhs_size, const TFloat* in_rhs, uint32_t rhs_size);

template<class TFloat>
void VectorMultiply(const TFloat* in_lhs, uint32_t vec_size, TFloat scalar, TFloat** out_vec);
}

#endif // _TEMPEST_MATRIX_VARIADIC_HH_

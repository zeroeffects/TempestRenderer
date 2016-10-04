#include "tempest/utils/testing.hh"
#include "tempest/math/matrix3.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/math/matrix-variadic.hh"

const uint32_t RandomSamples = 1024;

TGE_TEST("Testing matrix library")
{
    Tempest::Matrix3 basis_from_normal;
    basis_from_normal.makeBasis(Tempest::Vector3{0.0f, 0.0f, 1.0f});

    Tempest::Matrix3 identity;
    identity.identity();

    TGE_CHECK(basis_from_normal == identity, "Invalid basis construction");

    Tempest::Matrix3 basis_from_tangent;
    basis_from_tangent.makeBasisTangent(Tempest::Vector3{1.0f, 0.0f, 0.0f});

    TGE_CHECK(basis_from_tangent == identity, "Invalid basis construction");

    Tempest::Vector2 vec0{ 0.45f, 0.65f },
					 vec1{ -0.3f, 0.0f };

	auto len0 = Length(vec0);
	auto len1 = Length(vec1);

    vec0 /= len0;
    vec1 /= len1;

	auto mat = Tempest::Matrix2::rotation(vec0, vec1);

    auto rot_vec = mat.transform(vec0);

	TGE_CHECK(Tempest::Dot(rot_vec, vec1) > 0.9f, "Invalid rotation quaternion");

    auto rot_slerp_mat = mat.slerp(1.0f);

    TGE_CHECK(rot_slerp_mat == mat, "Broken slerp");

    auto rot_half_vec = mat.slerp(0.5f).transform(vec0);

    auto half_vec = vec0 + vec1;
    NormalizeSelf(&half_vec);

    TGE_CHECK(Tempest::Dot(half_vec, rot_half_vec) > 0.9f, "Bad slerp implementation");

    float t_test = 0.25f;
    auto sl_mat = Tempest::Matrix2::slerpLerpMatrix(vec0*len0, vec1*len1, t_test);

    auto predict_mat = mat.slerp(0.25f);
    auto rot_mat = predict_mat;
    predict_mat.scale((1.0f - t_test) + t_test*len1/len0);

    TGE_CHECK(predict_mat == sl_mat, "Invalid matrix");

    auto expect_vec = rot_mat.transform(vec0)*((1.0f - t_test) + t_test*len1/len0);
    auto transf_vec = sl_mat.transform(vec0);

    TGE_CHECK(expect_vec == transf_vec, "Invalid transformation vector");

    Tempest::Matrix3 cholesky_init_matrix({ 4.0f, 12.0f, -16.0f },
                                          { 12.0f, 37.0f, -43.0f },
                                          { -16.0f, -43.0f, 98.0f });

    Tempest::Matrix3 cholesky_matrix = cholesky_init_matrix.choleskyDecomposition();

    Tempest::Matrix3 recons_init_matrix = cholesky_matrix * cholesky_matrix.transpose();

    TGE_CHECK(recons_init_matrix == cholesky_init_matrix, "Invalid Cholesky decomposition");

    unsigned seed = 1;

    {
    auto variance = Tempest::Vector3{ 4.0f, 25.0f, 9.0f };

    Tempest::Matrix3 basis(Tempest::Vector3{ 0.0f, 1.0f, 2.0f },
                            Tempest::Vector3{ 3.0f, 4.0f, 5.0f },
                            Tempest::Vector3{ 6.0f, 7.0f, 10.0f });

    Tempest::Matrix3 scaled_basis(basis);
    scaled_basis.scale(variance);

    auto direct_cov_matrix = basis.transformCovariance(variance);

    Tempest::Matrix3 scale_indir(basis);
    scale_indir.scale(Tempest::Vector3Sqrt(variance));

    auto indir_matrix = scale_indir.transformCovariance(Tempest::ToVector3(1.0f));

    auto basis_transposed = basis.transpose();
    auto expected_cov_matrix = scaled_basis * basis_transposed;

    auto rev_expected_cov_matrix = basis_transposed * scaled_basis;
    TGE_CHECK(direct_cov_matrix == expected_cov_matrix, "Invalid multiplication with pre-transposed matrix");
    }

    Tempest::Matrix3 adj_test_mat({ 1.0f, 2.0f, 3.0f },
                                  { 4.0f, 5.0f, 6.0f },
                                  { 7.0f, 8.0f, 9.0f });

    auto adj_mat = adj_test_mat.adjugate();
    TGE_CHECK(adj_mat == Tempest::Matrix3({ -3.0f,   6.0f, -3.0f },
                                           {  6.0f, -12.0f,  6.0f },
                                           { -3.0f,   6.0f, -3.0f }), "Invalid adjugate");

	float generic_matrix[4] = { 1.0f, -1.0f, 1.0f, 1.0f };
	float out_matrix[4];
	float* out_matrix_ptr = out_matrix;

	auto status = Tempest::MatrixCholeskyDecomposition(generic_matrix, 2, &out_matrix_ptr);
	TGE_CHECK(!status, "Cholesky decomposition failure not indicated"); 

    float near = 0.1f, far = 1000.0f;
    const Tempest::Matrix4 proj = Tempest::PerspectiveMatrix(45.0f, 1920.0f / 1080.0f, near, far);

    for(size_t i = 0; i < RandomSamples; ++i)
    {
        auto dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Matrix3 basis;
        basis.makeBasis(dir);

        //*
        TGE_CHECK(Tempest::Dot(basis.tangent(), basis.normal()) < 1e-3f && Tempest::Dot(basis.binormal(), basis.normal()) < 1e-3f, "Non-ortho basis");

        Tempest::Matrix4 view(basis);
        auto view_proj = proj*view;
        
        auto view_proj_inverse = view_proj.inverse();

        auto view_transpose = basis.transpose();
        
        auto ident = view_transpose*basis;
        
        TGE_CHECK(ident == Tempest::Matrix3::identityMatrix(), "Not really identity matrix");

        auto norm = view_transpose.normal();
        
        auto origin = view_proj_inverse * Tempest::Vector3{ 0.0f, 0.0f, -1.0f };
        auto far_forward = view_proj_inverse * Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
        auto near_up = view_proj_inverse * Tempest::Vector3{ 0.0f, 1.0f, -1.0f };

        auto comp_origin = -near*norm;
        auto origin_screen = view_proj*comp_origin;

        TGE_CHECK(Tempest::ApproxEqual(comp_origin, origin, 1e-3f), "Invalid origin");
        
        auto forward = Tempest::Normalize(origin - far_forward);
        auto up = Tempest::Normalize(near_up - origin);
        float cos_data = fabsf(Tempest::Dot(forward, up));
        TGE_CHECK(cos_data < 1e-3f, "Non-ortho basis");
        auto left = Tempest::Normalize(Tempest::Cross(up, forward));
        
        auto near_down = view_proj_inverse * Tempest::Vector3{ 0.0f, -1.0f, -1.0f };
        auto down = Tempest::Normalize(origin - near_down);
        TGE_CHECK(up == down, "Unexpected dir");

        auto near_left = view_proj_inverse * Tempest::Vector3{ 1.0f, 0.0f, -1.0f };
        auto near_right = view_proj_inverse * Tempest::Vector3{ -1.0f, 0.0f, -1.0f };
        auto expected_left = Tempest::Normalize(near_left - origin);
        auto expected_right = Tempest::Normalize(origin - near_right);
        TGE_CHECK(expected_left == expected_right, "Invalid axis");
        TGE_CHECK(Tempest::Dot(up, expected_left) < 1e-3f && Tempest::Dot(forward, expected_left) < 1e-3f, "Non-ortho basis computed");
        TGE_CHECK(Tempest::Dot(up, left) < 1e-3f && Tempest::Dot(forward, left) < 1e-3f, "Non-ortho basis computed");

        auto dbg_origin = view_proj_inverse * Tempest::Vector3{ 0.0f, 0.0f, 0.0f };
        auto dbg_near_left = view_proj_inverse * Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
        auto dbg_left = Tempest::Normalize(dbg_near_left - dbg_origin);

        auto proj_inverse = proj.inverse();
        auto view_back = proj_inverse*view_proj;

        TGE_CHECK(view_back == view, "Invalid inverse transformation");

        auto quat_view_back = Tempest::Normalize(Tempest::ToQuaternion(view_back));
        auto quat_view = Tempest::Normalize(Tempest::ToQuaternion(basis));

        TGE_CHECK(Tempest::ApproxEqual(quat_view_back, quat_view, 1e-3f), "Unexpected view");

        auto view_result = view_proj_inverse.rotationFromPerspectiveInverseMatrix();
        TGE_CHECK(view_result.rotationMatrix3() == basis, "Invalid matrix");

        Tempest::Matrix3 recons_view(left, up, forward);
        recons_view = recons_view.transpose();

        auto quat_recons2_view = Tempest::Normalize(Tempest::ToQuaternion(recons_view));

        TGE_CHECK(Tempest::ApproxEqual(quat_recons2_view, quat_view, 1e-3f), "Invalid reconstruction");
        //*/

        auto dir2 = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        if(fabsf(Tempest::Dot(dir, dir2)) > 1e-3f)
        {
            Tempest::Matrix3 ortho_basis;
            ortho_basis.makeBasisOrthogonalize(dir, dir2);
            TGE_CHECK(Tempest::Dot(ortho_basis.tangent(), dir) > 1e-3f, "Invalid tangent vector");
        }


        Tempest::Vector3 stddev{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) };

        auto variance = stddev*stddev;

        Tempest::Matrix3 scaled_basis(basis);
        scaled_basis.scale(variance);

        auto direct_cov_matrix = basis.transformCovariance(variance);

        auto expected_cov_matrix = scaled_basis * basis.transpose();
        TGE_CHECK(expected_cov_matrix == expected_cov_matrix, "Invalid multiplication with pre-transposed matrix");

        Tempest::Matrix3 inv_basis = basis.inverse();
        TGE_CHECK(inv_basis == basis.transpose(), "Invalid inverse of rotation matrix");

        // Long-winded check of everything
        Tempest::Matrix3 scale_squared_matrix = Tempest::Matrix3::identityMatrix();
        scale_squared_matrix.scale(variance);

        Tempest::Matrix3 long_deriv_squared_matrix = basis*scale_squared_matrix*basis.transpose();
        TGE_CHECK(long_deriv_squared_matrix == expected_cov_matrix, "Different transformation matrix");

        auto second_dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        Tempest::Matrix3 actual_scaled_basis = basis;
        actual_scaled_basis.scale(stddev);

        Tempest::Matrix3 squared_matrix = actual_scaled_basis*actual_scaled_basis.transpose();

        TGE_CHECK(expected_cov_matrix == squared_matrix, "Unexpected covariance");

        float sq_transform = Length(actual_scaled_basis.transformRotationInverse(second_dir));

        float transform_sq_mat = sqrtf(Dot(second_dir, squared_matrix.transform(second_dir)));

        TGE_CHECK(Tempest::ApproxEqual(transform_sq_mat, sq_transform), "Unexpected final result after using squared matrix or transforming and then squaring");

        float full_equation_transform = sqrtf(Dot(second_dir, long_deriv_squared_matrix.transform(second_dir)));

        TGE_CHECK(Tempest::ApproxEqual(sq_transform, full_equation_transform), "Unexpected result after using the long decomposed set of matrices");
    }
    
    for(size_t i = 0; i < RandomSamples; ++i)
    {
        auto dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Matrix3 basis;
        basis.makeBasis(dir);

        Tempest::Vector3 stddev{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) };

        basis.scale(stddev);

        auto dir2 = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        auto direct = Normalize(basis.transform(dir2));

        auto basis_squared = basis*basis.transpose();

        auto cholesky_basis = basis_squared.choleskyDecomposition();

        auto restored_basis_squared = cholesky_basis*cholesky_basis.transpose();

        TGE_CHECK(basis_squared == restored_basis_squared, "Invalid Cholesky decomposition");

        auto indirect = Normalize(cholesky_basis.transform(dir2));

        auto cholesky_det = cholesky_basis.determinant();

        auto cholesky_transpose = cholesky_basis.transpose();
        auto cholesky_transpose_inverse = cholesky_transpose.inverse();

        auto expected_identity = cholesky_transpose*cholesky_transpose_inverse;
        TGE_CHECK(Tempest::ApproxEqual(expected_identity, Tempest::Matrix3::identityMatrix(), 1e-4f), "Inverse is broken");

        auto space_diff = basis_squared*cholesky_transpose_inverse;

        auto from_one_to_other = basis.transpose()*cholesky_transpose_inverse; // Obviously not much can be assumed about them

        TGE_CHECK(Tempest::ApproxEqual(space_diff, cholesky_basis, 1e-4f), "Mostly because space difference the vectors are not the same, but they share common covariance");
    }

    for(size_t i = 0; i < RandomSamples; ++i)
    {
        auto x_angle = 2.0f*Tempest::MathPi*Tempest::FastFloatRand(seed);
        auto y_angle = 2.0f*Tempest::MathPi*Tempest::FastFloatRand(seed);
        auto z_angle = 2.0f*Tempest::MathPi*Tempest::FastFloatRand(seed);

        Tempest::Matrix3 mat_x;
        mat_x.identity();
        mat_x.rotateX(x_angle);

        Tempest::Quaternion quat_x = Tempest::IdentityQuaternion();
        quat_x = Tempest::RotateX(quat_x, x_angle);

        Tempest::Matrix3 mat_quat_x = Tempest::ToMatrix3(quat_x);
        TGE_CHECK(mat_quat_x == mat_x, "Invalid rotation quaternion");

        Tempest::Matrix3 euler_rot_x;
        euler_rot_x.identity();
        euler_rot_x.rotate({ x_angle, 0.0f, 0.0f });

        TGE_CHECK(mat_x == euler_rot_x, "Invalid X rotation matrix");

        Tempest::Matrix3 mat_y;
        mat_y.identity();
        mat_y.rotateY(y_angle);

        Tempest::Quaternion quat_y = Tempest::IdentityQuaternion();
        quat_y = Tempest::RotateY(quat_y, y_angle);

        Tempest::Matrix3 mat_quat_y = Tempest::ToMatrix3(quat_y);
        TGE_CHECK(mat_quat_y == mat_y, "Invalid rotation quaternion");

        Tempest::Matrix3 euler_rot_y;
        euler_rot_y.identity();
        euler_rot_y.rotate({ 0.0f, y_angle, 0.0f });

        TGE_CHECK(mat_y == euler_rot_y, "Invalid Y rotation matrix");

        Tempest::Matrix3 mat_z;
        mat_z.identity();
        mat_z.rotateZ(z_angle);

        Tempest::Quaternion quat_z = Tempest::IdentityQuaternion();
        quat_z = Tempest::RotateZ(quat_z, z_angle);

        Tempest::Matrix3 mat_quat_z = Tempest::ToMatrix3(quat_z);
        TGE_CHECK(mat_quat_z == mat_z, "Invalid rotation quaternion");

        Tempest::Matrix3 euler_rot_z;
        euler_rot_z.identity();
        euler_rot_z.rotate({ 0.0f, 0.0f, z_angle });

        TGE_CHECK(mat_z == euler_rot_z, "Invalid Z rotation matrix");

        Tempest::Matrix3 mat_xy = mat_x*mat_y;

        Tempest::Matrix3 euler_rot_xy;
        euler_rot_xy.identity();
        euler_rot_xy.rotate({ x_angle, y_angle, 0.0f });

        TGE_CHECK(euler_rot_xy == mat_xy, "Invalid matrix based on Euler angles around x-y axis");

        Tempest::Matrix3 mat_xyz = mat_x*mat_y*mat_z;

        Tempest::Matrix3 euler_rot_xyz;
        euler_rot_xyz.identity();
        euler_rot_xyz.rotate({ x_angle, y_angle, z_angle });

        TGE_CHECK(euler_rot_xyz == mat_xyz, "Invalid matrix based on Euler angles around x-y axis");

        Tempest::Quaternion quat_xy = quat_x*quat_y;

        Tempest::Quaternion quat_xyz = quat_x*quat_y*quat_z;
        Tempest::Matrix3 mat_quat_xyz = Tempest::ToMatrix3(quat_xyz);

        TGE_CHECK(mat_quat_xyz == euler_rot_xyz, "Invalid quaternion rotation matrix from euler angles");

        Tempest::Quaternion quat_euler_xy = Tempest::ToQuaternion({ x_angle, y_angle, 0.0f });
        TGE_CHECK(quat_xy == quat_euler_xy, "Invalid quaternion from euler angles");

        Tempest::Quaternion quat_euler_xyz = Tempest::ToQuaternion({ x_angle, y_angle, z_angle });
        TGE_CHECK(quat_euler_xyz == quat_xyz, "Invalid quaternion from Euler angles");
    }

    for(size_t i = 0; i < RandomSamples; ++i)
    {
        auto dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Matrix3 basis;
        basis.makeBasis(dir);

        auto tan_quat = Tempest::ToQuaternionNormal(dir);
        auto mat_quat = Tempest::ToMatrix3(tan_quat);
        auto norm = mat_quat.normal();

        TGE_CHECK(Tempest::ApproxEqual(dir, norm, 1e-3f), "Invalid quaternion");

        Tempest::Vector3 scaling, euler;
        basis.decompose(&scaling, &euler);

        TGE_CHECK(Tempest::ApproxEqual(scaling, 1.0f, 1e-6f), "Invalid scaling decomposition. Matrix is supposed to be ortho-normal.");

        Tempest::Quaternion quat_from_mat = Tempest::ToQuaternion(basis);
        Tempest::Matrix3 recons_quat_basis = Tempest::ToMatrix3(quat_from_mat);
        TGE_CHECK(Tempest::ApproxEqual(recons_quat_basis, basis, 1e-3f), "Invalid quaternion conversion");

        Tempest::Quaternion quat = Tempest::ToQuaternion(euler);
        Tempest::Matrix3 recons_euler_quat_direct = Tempest::ToMatrix3(quat);
        TGE_CHECK(Tempest::ApproxEqual(basis, recons_euler_quat_direct, 1e-3f), "Invalid euler to quaternion conversion");

        /*
        auto recons_euler = Tempest::ToEulerAngles(quat);
        TGE_CHECK(Tempest::ApproxEqual(recons_euler, euler, 1e-3f), "Invalid euler angles reconstruction");

        Tempest::Matrix3 reconst_euler_quat_basis;
        reconst_euler_quat_basis.identity();
        reconst_euler_quat_basis.rotate(recons_euler);
        TGE_CHECK(Tempest::ApproxEqual(basis, reconst_euler_quat_basis, 1e-1f), "Invalid matrix basis");
        */

        Tempest::Matrix3 recons_basis;
        recons_basis.identity();
        recons_basis.rotate(euler);

        TGE_CHECK(Tempest::ApproxEqual(basis, recons_basis, 4e-2f), "Invalid matrix basis");
    }
}
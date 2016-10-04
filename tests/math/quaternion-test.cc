#include "tempest/utils/testing.hh" 
#include "tempest/math/quaternion.hh"
#include "tempest/math/sampling3.hh"

const uint32_t RandomSamples = 1000;

TGE_TEST("Quaternion test")
{
    unsigned seed = 1;
    for(uint32_t i = 0; i < RandomSamples; ++i)
    {
        Tempest::Vector3 vec0 = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Vector3 vec1 = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));


        Tempest::Matrix3 mat_bas;
        mat_bas.makeBasis(vec0);

        auto quat_bas = Tempest::ToQuaternion(mat_bas);
        auto tan_quat_bas = Tempest::Transform(quat_bas, Tempest::Vector3{ 1.0f, 0.0f, 0.0f });
        auto binorm_quat_bas = Tempest::Transform(quat_bas, Tempest::Vector3{ 0.0f, 1.0f, 0.0f });
        auto norm_quat_bas = Tempest::Transform(quat_bas, Tempest::Vector3{ 0.0f, 0.0f, 1.0f });

        auto recons_mat_bas = Tempest::ToMatrix3(quat_bas);

        TGE_CHECK(Tempest::ApproxEqual(tan_quat_bas, mat_bas.tangent(), 1e-3f) &&
                   Tempest::ApproxEqual(binorm_quat_bas, mat_bas.binormal(), 1e-3f) &&
                   Tempest::ApproxEqual(norm_quat_bas, mat_bas.normal(), 1e-3f), "Invalid matrix to quaternion conversion");

        TGE_CHECK(Tempest::ApproxEqual(mat_bas, recons_mat_bas, 1e-3f), "Invalid conversion back from quaternion");

        TGE_CHECK(Tempest::ApproxEqual(Length(vec0), 1.0f, 1e-3f), "Invalid sample");
        TGE_CHECK(Tempest::ApproxEqual(Length(vec1), 1.0f, 1e-3f), "Invalid sample");

        float dot_vec = Tempest::Dot(vec0, vec1);
        if(dot_vec < -0.9f)
            continue;

	    auto quat = Tempest::FastRotationBetweenVectorQuaternion(vec0, vec1);

        auto cons_quat = Tempest::ConservativeRotationBetweenVectorQuaternion(vec0, vec1);

        TGE_CHECK(Tempest::ApproxEqual(quat, cons_quat, 1e-3f), "Invalid rotation quaternion computations");

        float angle = acosf(dot_vec);
        auto axis = Normalize(Tempest::Cross(vec0, vec1));

        Tempest::Matrix3 mat3_from_axis(angle, axis);

        float dot_norm_tan = Dot(mat3_from_axis.normal(), mat3_from_axis.tangent());
        float dot_norm_binorm = Dot(mat3_from_axis.normal(), mat3_from_axis.binormal());

        TGE_CHECK(Tempest::ApproxEqual(dot_norm_tan, 0.0f, 1e-3f) &&
                   Tempest::ApproxEqual(dot_norm_binorm, 0.0f, 1e-3f), "Not orthogonal basis");

        auto quat_through_mat3 = Tempest::ToQuaternion(mat3_from_axis);

        auto tan_quat = Tempest::Transform(quat_through_mat3, Tempest::Vector3{ 1.0f, 0.0f, 0.0f });
        auto binorm_quat = Tempest::Transform(quat_through_mat3, Tempest::Vector3{ 0.0f, 1.0f, 0.0f });
        auto norm_quat = Tempest::Transform(quat_through_mat3, Tempest::Vector3{ 0.0f, 0.0f, 1.0f });
       
        TGE_CHECK(Tempest::ApproxEqual(tan_quat, mat3_from_axis.tangent(), 1e-3f) &&
                   Tempest::ApproxEqual(binorm_quat, mat3_from_axis.binormal(), 1e-3f) &&
                   Tempest::ApproxEqual(norm_quat, mat3_from_axis.normal(), 1e-3f), "Invalid matrix to quaternion conversion");

        Tempest::Matrix4 mat4_from_axis;
        mat4_from_axis.identity();
        mat4_from_axis.rotate(angle, axis);

        auto quat_through_mat4 = Tempest::ToQuaternion(mat4_from_axis);

        TGE_CHECK(quat_through_mat3 == quat_through_mat4, "Invalid matrix4 conversion from quaternion");

        auto recons_mat3 = Tempest::ToMatrix3(quat_through_mat3);
        Tempest::Matrix4 recons_mat4(quat_through_mat4, {});

        TGE_CHECK(Tempest::ApproxEqual(recons_mat3, mat3_from_axis, 1e-3f), "Invalid conversion back from quaternion");

	    auto rot_vec = Tempest::Transform(quat, vec0);

	    TGE_CHECK(Tempest::Dot(rot_vec, vec1) > 0.9f, "Invalid rotation quaternion");

        auto rot_slerp_quat = Tempest::Slerp(quat, 1.0f);

        TGE_CHECK(Tempest::ApproxEqual(rot_slerp_quat, quat, 1e-3f), "Broken slerp");

        auto rot_half_vec = Tempest::Transform(Tempest::Slerp(quat, 0.5f), vec0);

        auto half_vec = vec0 + vec1;
        NormalizeSelf(&half_vec);

        TGE_CHECK(Tempest::Dot(half_vec, rot_half_vec) > 0.9f, "Bad slerp implementation");
    }
}
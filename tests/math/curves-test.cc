#include "tempest/utils/testing.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/curves.hh"

const size_t Samples = 1000;

TGE_TEST("Testing spline curves")
{
    Tempest::Vector2 cm_coef[4]
    {
        { 0.0f, 3.0f },
        { 0.25f, 1.0f },
        { 0.5f, 2.0f },
        { 0.75f, 0.0f }
    };

    auto p1_recons = Tempest::CatmullRomCurvePoint(cm_coef[0], cm_coef[1], cm_coef[2], cm_coef[3], 0.0f);
    TGE_CHECK(Tempest::Length(p1_recons - cm_coef[1]) < 1e-6f, "Invalid reconstruction");

    auto p2_recons = Tempest::CatmullRomCurvePoint(cm_coef[0], cm_coef[1], cm_coef[2], cm_coef[3], 1.0f);
    TGE_CHECK(Tempest::Length(p2_recons - cm_coef[2]) < 1e-6f, "Invalid reconstruction");

    auto p1_matrix_recons = Tempest::EvaluateCurve(Tempest::CatmullRomCurveMatrix, cm_coef, 0.0f);
    TGE_CHECK(Tempest::Length(p1_recons - cm_coef[1]) < 1e-6f, "Invalid reconstruction");

    auto p2_matrix_recons = Tempest::EvaluateCurve(Tempest::CatmullRomCurveMatrix, cm_coef, 1.0f);
    TGE_CHECK(Tempest::Length(p2_recons - cm_coef[2]) < 1e-6f, "Invalid reconstruction");

    decltype(cm_coef) bezier_coef;
    Tempest::CatmullRomToBezier(cm_coef, bezier_coef);

    decltype(cm_coef) recons_cm_coef;
    Tempest::BezierToCatmullRom(bezier_coef, recons_cm_coef);

    for(size_t i = 0; i < 4; ++i)
    {
        TGE_CHECK(Tempest::Length(cm_coef[i] - recons_cm_coef[i]) < 1e-6f, "Invalid inverse transformation from Bezier to Catmull-Rom");
    }

    Tempest::Matrix4 check_inv0 = Tempest::BezierCurveMatrix.inverse()*Tempest::BezierCurveMatrix;
    TGE_CHECK(check_inv0 == Tempest::Matrix4::identityMatrix(), "Invalid inverse matrix");

    Tempest::Matrix4 check_inv1 = Tempest::CatmullRomCurveMatrix.inverse()*Tempest::CatmullRomCurveMatrix;
    TGE_CHECK(check_inv1 == Tempest::Matrix4::identityMatrix(), "Invalid inverse matrix");

    Tempest::Matrix4 cor = Tempest::BezierCurveMatrix.inverse()*Tempest::CatmullRomCurveMatrix;
    auto cm_coef_y = Tempest::Vector4{ cm_coef[0].y, cm_coef[1].y, cm_coef[2].y, cm_coef[3].y };
    auto bezier_manual_coef = cor * Tempest::Vector4{ cm_coef[0].y, cm_coef[1].y, cm_coef[2].y, cm_coef[3].y };
    TGE_CHECK(Tempest::Length(Tempest::Vector4{ bezier_coef[0].y, bezier_coef[1].y, bezier_coef[2].y, bezier_coef[3].y } - bezier_manual_coef), "Invalid bezier curve");
    auto p0_bezier_recons = Tempest::BezierCurveMatrix*bezier_manual_coef;
    auto p0_cm_recons = Tempest::CatmullRomCurveMatrix*cm_coef_y;
    TGE_CHECK(Tempest::Length4D(p0_bezier_recons - p0_cm_recons) < 1e-6f, "Invalid reconstruction");

    auto p1_bezier_recons = Tempest::EvaluateCurve(Tempest::BezierCurveMatrix, bezier_coef, 0.0f);
    auto p2_bezier_recons = Tempest::EvaluateCurve(Tempest::BezierCurveMatrix, bezier_coef, 1.0f);
    TGE_CHECK(Tempest::Length(p1_bezier_recons - cm_coef[1]) < 1e-6f, "Invalid reconstruction");
    TGE_CHECK(Tempest::Length(p2_bezier_recons - cm_coef[2]) < 1e-6f, "Invalid reconstruction");

    for(size_t i = 0; i < Samples; ++i)
    {
        float t = (float)i/(Samples - 1);

        auto bezier_p1 = Tempest::EvaluateCurve(Tempest::BezierCurveMatrix, bezier_coef, t);
        auto cm_p1 = Tempest::EvaluateCurve(Tempest::CatmullRomCurveMatrix, cm_coef, t);
        TGE_CHECK(Tempest::Length(bezier_p1 - cm_p1) < 1e-6f, "Invalid curve conversion");

        auto bezier_p2 = Tempest::BezierCurvePoint(bezier_coef[0], bezier_coef[1], bezier_coef[2], bezier_coef[3], t);
        auto cm_p2 = Tempest::CatmullRomCurvePoint(cm_coef[0], cm_coef[1], cm_coef[2], cm_coef[3], t);
        TGE_CHECK(Tempest::Length(bezier_p2 - cm_p2) < 1e-6f, "Invalid curve conversion");

        auto bezier_manual_p = Tempest::BezierCurvePoint(bezier_manual_coef.x, bezier_manual_coef.y, bezier_manual_coef.z, bezier_manual_coef.w, t);
        TGE_CHECK(Tempest::Length(cm_p2.y - bezier_manual_p) < 1e-6f, "Invalid curve conversion");
    }

}
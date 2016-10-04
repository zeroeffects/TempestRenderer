#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"

const uint32_t IntegratorSamples = 1024;

TGE_TEST("Testing fundamental properties of BRDF functions")
{
    Tempest::IlluminationModel illum_models[] =
    {
        Tempest::IlluminationModel::KajiyaKay,
        Tempest::IlluminationModel::BlinnPhong,
        Tempest::IlluminationModel::AshikhminShirley,
    //    Tempest::IlluminationModel::GGXMicrofacet,
        Tempest::IlluminationModel::BeckmannMicrofacet,
    };

    Tempest::TransmittanceFunction tested_trans_funcs[] =
    {
        Tempest::KajiyaKayBRDF,
        Tempest::BlinnPhongBRDF,
        Tempest::AshikhminShirleyBRDF,
    //    Tempest::GGXMicrofacetBRDF,
        Tempest::BeckmannMicrofacetBRDF,
    };

    Tempest::PDFFunction tested_pdfs[] =
    {
        Tempest::UniformHemispherePDF,
        Tempest::BlinnPhongPDF,
        Tempest::AshikhminShirleyPDF,
    //    Tempest::GGXMicrofacetPDF,
        Tempest::BeckmannMicrofacetPDF,
    };

    Tempest::RTMicrofacetMaterial material;
    material.Specular = Tempest::ToSpectrum(1.0f);
    material.SpecularPower = { 10.0f, 20.0f };
    material.Diffuse = {};
    material.Fresnel = { 1.0f, 0.0f };
    material.setup();

    Tempest::SampleData sample_data;
    sample_data.Material = &material;
    sample_data.OutgoingLight = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
    sample_data.Tangent =  Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    sample_data.Normal =   Tempest::Vector3{ 0.0f, 0.0f, 1.0f };

    for(auto tested_pdf : tested_pdfs)
    {
        float pdf_total = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntegratorSamples,
                                [tested_pdf, &sample_data](const Tempest::Vector3& dir)
                                {   
                                    sample_data.IncidentLight = dir;

                                    float pdf = tested_pdf(sample_data);

                                    TGE_CHECK(pdf >= 0.0f, "Invalid PDF");

                                    return pdf;
                                });

        TGE_CHECK(Tempest::ApproxEqual(pdf_total, 1.0f, 1e-1f), "Invalid PDF");
    }

    size_t i = 0;
    for(auto tested_transmittance : tested_trans_funcs)
    {
        material.Model = illum_models[i++];
        material.setup();

        Tempest::Spectrum total_contrib = Tempest::StratifiedMonteCarloIntegratorHemisphere<Tempest::Spectrum>(IntegratorSamples,
                                                [tested_transmittance, &sample_data](const Tempest::Vector3& dir)
                                                {   
                                                    sample_data.IncidentLight = dir;

                                                    auto brdf = tested_transmittance(sample_data);

                                                    TGE_CHECK(Array(brdf)[0] >= 0.0f, "Invalid PDF");

                                                    return brdf;
                                                });

        TGE_CHECK(Array(total_contrib)[0] <= 1.1f, "Invalid transmittance function"); // Complete energy conservation is hard to guarantee, but energy generation is not tolerated
    }
}
#include "tempest/utils/testing.hh"
#include "tempest/math/intersect.hh"
#include "tempest/utils/viewer.hh"

TGE_TEST("Testing intersection functions")
{
    float t, u, v;
    Tempest::Vector3 norm;

	Tempest::Vector2 v0{ -2.0f, 1.0f },
					 v1{ 2.0f, 1.0f },
					 v2{ 0.0f, 2.0f };

	Tempest::Vector2 tc0, tc1, tc2;
	Tempest::TriangleBarycentricCoordinates(v1, v0, v1, v2, &tc1);
	TGE_CHECK(Tempest::ApproxEqual(tc1.x, 1.0f), "Unexpected barycentric coordinate");

	Tempest::TriangleBarycentricCoordinates(v2, v0, v1, v2, &tc2);
	TGE_CHECK(Tempest::ApproxEqual(tc2.y, 1.0f), "Unexpected barycentric coordinate");

	Tempest::TriangleBarycentricCoordinates(v0, v0, v1, v2, &tc0);
	TGE_CHECK(Tempest::ApproxEqual(1.0f - tc0.x - tc0.y, 1.0f), "Unexpected barycentric coordinate");

    const float sphere_radius = 1.0f;

    Tempest::Vector3 org{ 0.0f, 0.0f, 0.0f };
    Tempest::Vector3 dir{ 0.0f, 0.0f, 1.0f };

    Tempest::Sphere sphere;
    sphere.Center = dir*sphere_radius*1.1f;
    sphere.Radius = sphere_radius;


    bool intersect = Tempest::IntersectSphere(dir, org, sphere, &t, &u, &v, &norm);
    TGE_CHECK(intersect, "Failed to perform sphere intersection");

    intersect = Tempest::IntersectSphere(dir, org + Tempest::Vector3{ sphere_radius, sphere_radius, 0.0f}, sphere, &t, &u, &v, &norm);
    TGE_CHECK(!intersect, "False positive intersection");

    sphere.Center = {};
    sphere.Radius = 0.5f;

    Tempest::AABBUnaligned box;
    box.MinCorner = { -1.0f, -1.0f, -1.0f };
    box.MaxCorner = { 1.0f, 1.0f, 1.0f };

    intersect = Tempest::IntersectSphereAABB(sphere, box);
    TGE_CHECK(intersect, "Broken sphere box intersection");

    intersect = Tempest::IntersectSphereCenteredAABB(sphere, box.MaxCorner);
    TGE_CHECK(intersect, "Broken sphere box intersection");

    {
    Tempest::Ellipsoid el0;
    el0.Center = dir*sphere_radius*1.1f;
    el0.Orientation = Tempest::IdentityQuaternion();
    el0.Scale = { sphere_radius, sphere_radius, sphere_radius };

    intersect = Tempest::IntersectEllipsoid(dir, org, el0, &t, &u, &v, &norm);
    TGE_CHECK(intersect, "Failed to perform sphere intersection");

    intersect = Tempest::IntersectEllipsoid(dir, org + Tempest::Vector3{ sphere_radius, sphere_radius, 0.0f}, el0, &t, &u, &v, &norm);
    TGE_CHECK(!intersect, "False positive intersection");
    }

    {
    Tempest::Ellipsoid el1;
    el1.Center = dir*sphere_radius*1.1f;
    el1.Orientation = Tempest::IdentityQuaternion();
    el1.Scale = { sphere_radius, 0.5f*sphere_radius, sphere_radius };

    intersect = Tempest::IntersectEllipsoid(dir, org + Tempest::Vector3{ 0.75f*sphere_radius, 0.0f, 0.0f}, el1, &t, &u, &v, &norm);
    TGE_CHECK(intersect, "Failed to intersect ellipsoid intersection");

    intersect = Tempest::IntersectEllipsoid(dir, org + Tempest::Vector3{ 0.0f, 0.75f*sphere_radius, 0.0f}, el1, &t, &u, &v, &norm);
    TGE_CHECK(!intersect, "False positive intersection");
    }

    {
    Tempest::Ellipsoid el1;
    el1.Center = dir*sphere_radius*1.1f;
    el1.Orientation = Tempest::RotateZ(Tempest::IdentityQuaternion(), Tempest::MathPi*0.5f);
    el1.Scale = { sphere_radius, 0.5f*sphere_radius, sphere_radius };

    intersect = Tempest::IntersectEllipsoid(dir, org + Tempest::Vector3{ 0.0f, 0.75f*sphere_radius, 0.0f}, el1, &t, &u, &v, &norm);
    TGE_CHECK(intersect, "Failed to intersect ellipsoid intersection");

    intersect = Tempest::IntersectEllipsoid(dir, org + Tempest::Vector3{ 0.75f*sphere_radius, 0.0f, 0.0f}, el1, &t, &u, &v, &norm);
    TGE_CHECK(!intersect, "False positive intersection");
    }

    {
    Tempest::Ellipsoid el2;
    el2.Center = dir*sphere_radius*1.1f;
    el2.Orientation = Tempest::RotateY(Tempest::IdentityQuaternion(), Tempest::MathPi*0.5f);
    el2.Scale = { sphere_radius, sphere_radius, 0.5f*sphere_radius };

    intersect = Tempest::IntersectEllipsoid(dir, org + Tempest::Vector3{ 0.0f, 0.75f*sphere_radius, 0.0f}, el2, &t, &u, &v, &norm);
    TGE_CHECK(intersect, "False positive intersection");

    intersect = Tempest::IntersectEllipsoid(dir, org + Tempest::Vector3{ 0.75f*sphere_radius, 0.0f, 0.0f}, el2, &t, &u, &v, &norm);
    TGE_CHECK(!intersect, "False positive intersection");
    }

    // That's a minimal repro case of a buggy code
    {
    uint32_t image_width = 200, image_height = 200;

    Tempest::FreeCamera cam;
	cam.Yaw = 0.0f;
	cam.Roll = Tempest::ToRadians(45.0f);
    cam.Offset = 10.0f;
    cam.Projection = Tempest::PerspectiveMatrix(40.0f, (float)image_width/image_height, 0.1f, 1000.0f);

    Tempest::Matrix4 view_proj_inv = Tempest::ComputeViewProjectionInverse(cam);
    
    Tempest::Ellipsoid el4;
    el4.Center = { 0.0f, 0.0f, 0.0f };
    el4.Scale = Tempest::Vector3{ 2.33876181f, 4.47130537f, 5.0f }/5.0f;
    el4.Orientation = { 0.00239738449f, 0.121537216f, -0.339387745f, 0.932758689f };
    
    Tempest::Sphere sphere;
    sphere.Center = { 0.0f, 0.0f, 0.0f };
    sphere.Radius = 1.1f;

    bool ellipsoid_intersect = false,
         sphere_intersect = false;
    for(uint32_t y = 0; y < image_height; ++y)
        for(uint32_t x = 0; x < image_width; ++x)
        {
            Tempest::Vector4 screen_tc = Tempest::Vector4{2.0f*x/(image_width - 1) - 1.0f, 2.0f*y/(image_height - 1) - 1.0f, -1.0f, 1.0};

			Tempest::Vector4 pos_start = view_proj_inv*screen_tc;

			screen_tc.z = 1.0f;
			Tempest::Vector4 pos_end = view_proj_inv*screen_tc;

			auto start_ray_pos = Tempest::ToVector3(pos_start);
			auto end_ray_pos = Tempest::ToVector3(pos_end);

			auto inc_light = Normalize(end_ray_pos - start_ray_pos);

            sphere_intersect |= Tempest::IntersectSphere(inc_light, start_ray_pos, sphere, &t, &u, &v, &norm);

            ellipsoid_intersect |= Tempest::IntersectEllipsoid(inc_light, start_ray_pos, el4, &t, &u, &v, &norm);

			TGE_CHECK(ellipsoid_intersect && sphere_intersect || !ellipsoid_intersect, "Invalid intersection");
        }

    TGE_CHECK(sphere_intersect, "Failed to intersect sphere");
    TGE_CHECK(ellipsoid_intersect, "Broken ray tracing code");
    }
}
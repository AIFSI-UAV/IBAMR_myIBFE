// ---------------------------------------------------------------------
//
// Copyright (c) 2017 - 2025 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

#include <SAMRAI_config.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/fe_base.h>
#include <libmesh/mesh.h>
#include <libmesh/quadrature_gauss.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <boost/multi_array.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <ibamr/app_namespaces.h>

namespace ModelData
{
enum class SwimmerStage
{
    PENALTY_PRESCRIBED = 1,
    ACTIVE_STRAIN = 2
};

struct ReferenceGeometry
{
    VectorValue<double> tangent;
    VectorValue<double> normal;
    libMesh::Point head;
    double length = 0.0;
};

struct TargetState
{
    libMesh::Point position;
    VectorValue<double> velocity;
};

struct SwimmerData
{
    SwimmerStage stage;
    ReferenceGeometry geometry;

    double frequency;
    double wave_count;
    double phase0;
    double ramp_time;

    double prescribed_head_amplitude;
    double prescribed_tail_amplitude;
    double prescribed_envelope_power;
    double penalty_stiffness;
    double penalty_damping;

    double passive_mu;
    double passive_lambda;
    double passive_fiber_modulus;
    double passive_shear_modulus;

    double active_curvature_tail;
    double active_envelope_power;
    double active_stretch_min;
    double active_stretch_max;

    SwimmerData(Pointer<Database> input_db, const ReferenceGeometry& reference_geometry)
        : stage(static_cast<SwimmerStage>(input_db->getInteger("SWIMMER_STAGE"))),
          geometry(reference_geometry),
          frequency(input_db->getDouble("WAVE_FREQUENCY")),
          wave_count(input_db->getDouble("WAVE_COUNT")),
          phase0(input_db->getDoubleWithDefault("WAVE_PHASE0", 0.0)),
          ramp_time(input_db->getDouble("WAVE_RAMP_TIME")),
          prescribed_head_amplitude(input_db->getDouble("PRESCRIBED_HEAD_AMPLITUDE_OVER_L") *
                                    reference_geometry.length),
          prescribed_tail_amplitude(input_db->getDouble("PRESCRIBED_TAIL_AMPLITUDE_OVER_L") *
                                    reference_geometry.length),
          prescribed_envelope_power(input_db->getDouble("PRESCRIBED_ENVELOPE_POWER")),
          penalty_stiffness(input_db->getDouble("PENALTY_STIFFNESS")),
          penalty_damping(input_db->getDouble("PENALTY_DAMPING")),
          passive_mu(input_db->getDouble("PASSIVE_MU")),
          passive_lambda(input_db->getDouble("PASSIVE_LAMBDA")),
          passive_fiber_modulus(input_db->getDouble("PASSIVE_FIBER_MODULUS")),
          passive_shear_modulus(input_db->getDouble("PASSIVE_SHEAR_MODULUS")),
          active_curvature_tail(input_db->getDouble("ACTIVE_CURVATURE_TAIL_TIMES_L") /
                                reference_geometry.length),
          active_envelope_power(input_db->getDouble("ACTIVE_ENVELOPE_POWER")),
          active_stretch_min(input_db->getDouble("ACTIVE_STRETCH_MIN")),
          active_stretch_max(input_db->getDouble("ACTIVE_STRETCH_MAX"))
    {
        const int stage_number = static_cast<int>(stage);
        if (stage_number != 1 && stage_number != 2)
        {
            TBOX_ERROR("SWIMMER_STAGE must be 1 (penalty prescribed) or 2 (active strain).\n");
        }
        if (geometry.length <= 0.0) TBOX_ERROR("The reference mesh has nonpositive body length.\n");
        if (frequency <= 0.0) TBOX_ERROR("WAVE_FREQUENCY must be positive.\n");
        if (wave_count <= 0.0) TBOX_ERROR("WAVE_COUNT must be positive.\n");
        if (ramp_time < 0.0) TBOX_ERROR("WAVE_RAMP_TIME must be nonnegative.\n");
        if (prescribed_head_amplitude < 0.0 || prescribed_tail_amplitude < 0.0)
        {
            TBOX_ERROR("Prescribed amplitudes must be nonnegative.\n");
        }
        if (prescribed_envelope_power < 1.0)
        {
            TBOX_ERROR("PRESCRIBED_ENVELOPE_POWER must be at least 1.\n");
        }
        if (penalty_stiffness < 0.0 || penalty_damping < 0.0)
        {
            TBOX_ERROR("Penalty coefficients must be nonnegative.\n");
        }
        if (stage == SwimmerStage::PENALTY_PRESCRIBED && penalty_stiffness <= 0.0)
        {
            TBOX_ERROR("Stage 1 requires PENALTY_STIFFNESS > 0.\n");
        }
        if (passive_mu <= 0.0 || passive_lambda < 0.0 || passive_fiber_modulus < 0.0 ||
            passive_shear_modulus < 0.0)
        {
            TBOX_ERROR("Passive material coefficients are invalid.\n");
        }
        if (active_envelope_power < 0.0)
        {
            TBOX_ERROR("ACTIVE_ENVELOPE_POWER must be nonnegative.\n");
        }
        if (!(0.0 < active_stretch_min && active_stretch_min < 1.0 && 1.0 < active_stretch_max))
        {
            TBOX_ERROR("Require 0 < ACTIVE_STRETCH_MIN < 1 < ACTIVE_STRETCH_MAX.\n");
        }
    }
};

double
dot2(const VectorValue<double>& a, const VectorValue<double>& b)
{
    return a(0) * b(0) + a(1) * b(1);
}

double
reference_s(const libMesh::Point& X, const SwimmerData& data)
{
    const VectorValue<double> displacement = X - data.geometry.head;
    return std::clamp(dot2(displacement, data.geometry.tangent), 0.0, data.geometry.length);
}

double
reference_eta(const libMesh::Point& X, const SwimmerData& data)
{
    const VectorValue<double> displacement = X - data.geometry.head;
    return dot2(displacement, data.geometry.normal);
}

double
unit_coordinate(const double s, const SwimmerData& data)
{
    return std::clamp(s / data.geometry.length, 0.0, 1.0);
}

void
ramp(const double time, const SwimmerData& data, double& value, double& derivative)
{
    if (data.ramp_time <= 0.0 || time >= data.ramp_time)
    {
        value = 1.0;
        derivative = 0.0;
    }
    else if (time <= 0.0)
    {
        value = 0.0;
        derivative = 0.0;
    }
    else
    {
        const double argument = M_PI * time / data.ramp_time;
        value = 0.5 * (1.0 - std::cos(argument));
        derivative = 0.5 * M_PI / data.ramp_time * std::sin(argument);
    }
}

double
wave_number(const SwimmerData& data)
{
    return 2.0 * M_PI * data.wave_count / data.geometry.length;
}

double
angular_frequency(const SwimmerData& data)
{
    return 2.0 * M_PI * data.frequency;
}

void
prescribed_amplitude(const double s, const SwimmerData& data, double& amplitude, double& derivative)
{
    const double xi = unit_coordinate(s, data);
    const double delta = data.prescribed_tail_amplitude - data.prescribed_head_amplitude;
    amplitude = data.prescribed_head_amplitude + delta * std::pow(xi, data.prescribed_envelope_power);
    derivative = delta * data.prescribed_envelope_power *
                 std::pow(xi, data.prescribed_envelope_power - 1.0) / data.geometry.length;
}

void
target_angle(const double s,
             const double time,
             const SwimmerData& data,
             double& theta,
             double& theta_time)
{
    double ramp_value = 0.0;
    double ramp_derivative = 0.0;
    ramp(time, data, ramp_value, ramp_derivative);

    double amplitude = 0.0;
    double amplitude_derivative = 0.0;
    prescribed_amplitude(s, data, amplitude, amplitude_derivative);

    const double k = wave_number(data);
    const double omega = angular_frequency(data);
    const double phase = k * s - omega * time + data.phase0;
    const double spatial_wave =
        amplitude_derivative * std::sin(phase) + amplitude * k * std::cos(phase);
    const double spatial_wave_time =
        -omega * amplitude_derivative * std::cos(phase) + omega * amplitude * k * std::sin(phase);
    const double slope = ramp_value * spatial_wave;
    const double slope_time = ramp_derivative * spatial_wave + ramp_value * spatial_wave_time;

    theta = std::atan(slope);
    theta_time = slope_time / (1.0 + slope * slope);
}

TargetState
target_state(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    static const std::array<double, 8> gauss_points = { -0.9602898564975363,
                                                        -0.7966664774136267,
                                                        -0.5255324099163290,
                                                        -0.1834346424956498,
                                                        0.1834346424956498,
                                                        0.5255324099163290,
                                                        0.7966664774136267,
                                                        0.9602898564975363 };
    static const std::array<double, 8> gauss_weights = { 0.1012285362903763,
                                                         0.2223810344533745,
                                                         0.3137066458778873,
                                                         0.3626837833783620,
                                                         0.3626837833783620,
                                                         0.3137066458778873,
                                                         0.2223810344533745,
                                                         0.1012285362903763 };

    const double s = reference_s(X, data);
    const double eta = reference_eta(X, data);
    double center_x = 0.0;
    double center_y = 0.0;
    double center_velocity_x = 0.0;
    double center_velocity_y = 0.0;

    for (std::size_t q = 0; q < gauss_points.size(); ++q)
    {
        const double sigma = 0.5 * s * (gauss_points[q] + 1.0);
        double theta = 0.0;
        double theta_time = 0.0;
        target_angle(sigma, time, data, theta, theta_time);
        const double weight = 0.5 * s * gauss_weights[q];
        center_x += weight * std::cos(theta);
        center_y += weight * std::sin(theta);
        center_velocity_x -= weight * std::sin(theta) * theta_time;
        center_velocity_y += weight * std::cos(theta) * theta_time;
    }

    double theta = 0.0;
    double theta_time = 0.0;
    target_angle(s, time, data, theta, theta_time);
    const double normal_x = -std::sin(theta);
    const double normal_y = std::cos(theta);
    const double normal_velocity_x = -std::cos(theta) * theta_time;
    const double normal_velocity_y = -std::sin(theta) * theta_time;

    const double local_x = center_x + eta * normal_x;
    const double local_y = center_y + eta * normal_y;
    const double local_velocity_x = center_velocity_x + eta * normal_velocity_x;
    const double local_velocity_y = center_velocity_y + eta * normal_velocity_y;

    TargetState target;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        target.position(d) = data.geometry.head(d) + local_x * data.geometry.tangent(d) +
                             local_y * data.geometry.normal(d);
        target.velocity(d) =
            local_velocity_x * data.geometry.tangent(d) + local_velocity_y * data.geometry.normal(d);
    }
    return target;
}

double
active_curvature(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    const double s = reference_s(X, data);
    const double xi = unit_coordinate(s, data);
    double ramp_value = 0.0;
    double ramp_derivative = 0.0;
    ramp(time, data, ramp_value, ramp_derivative);
    const double phase = wave_number(data) * s - angular_frequency(data) * time + data.phase0;
    return ramp_value * data.active_curvature_tail * std::pow(xi, data.active_envelope_power) *
           std::sin(phase);
}

double
active_stretch(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    if (data.stage != SwimmerStage::ACTIVE_STRAIN) return 1.0;
    const double stretch = 1.0 - reference_eta(X, data) * active_curvature(X, time, data);
    return std::clamp(stretch, data.active_stretch_min, data.active_stretch_max);
}

TensorValue<double>
dyad(const VectorValue<double>& a, const VectorValue<double>& b)
{
    TensorValue<double> result;
    result.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j) result(i, j) = a(i) * b(j);
    }
    return result;
}

void
PK1_material_function(TensorValue<double>& PP,
                      const TensorValue<double>& FF,
                      const libMesh::Point& /*x*/,
                      const libMesh::Point& X,
                      Elem* const /*elem*/,
                      const vector<const vector<double>*>& /*var_data*/,
                      const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                      const double time,
                      void* ctx)
{
    const auto& data = *static_cast<SwimmerData*>(ctx);
    const double lambda_active = active_stretch(X, time, data);
    const double J_active = lambda_active;

    const TensorValue<double> tt = dyad(data.geometry.tangent, data.geometry.tangent);
    const TensorValue<double> nn = dyad(data.geometry.normal, data.geometry.normal);
    const TensorValue<double> Fa_inverse = (1.0 / lambda_active) * tt + nn;

    TensorValue<double> Fe;
    Fe.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            for (unsigned int k = 0; k < NDIM; ++k) Fe(i, j) += FF(i, k) * Fa_inverse(k, j);
        }
    }

    const double J_elastic = Fe(0, 0) * Fe(1, 1) - Fe(0, 1) * Fe(1, 0);
    if (!(J_elastic > 0.0) || !std::isfinite(J_elastic))
    {
        TBOX_ERROR("Nonpositive elastic Jacobian in active-strain material at time "
                   << time << ", X=(" << X(0) << "," << X(1) << "), Je=" << J_elastic << "\n");
    }

    const TensorValue<double> Fe_inverse_transpose = tensor_inverse_transpose(Fe, NDIM);
    TensorValue<double> Pe;
    Pe.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            Pe(i, j) = data.passive_mu * (Fe(i, j) - Fe_inverse_transpose(i, j)) +
                       data.passive_lambda * std::log(J_elastic) * Fe_inverse_transpose(i, j);
        }
    }

    VectorValue<double> fiber;
    VectorValue<double> transverse;
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        fiber(i) = 0.0;
        transverse(i) = 0.0;
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            fiber(i) += Fe(i, j) * data.geometry.tangent(j);
            transverse(i) += Fe(i, j) * data.geometry.normal(j);
        }
    }

    const double fiber_stretch = std::sqrt(dot2(fiber, fiber));
    if (fiber_stretch > 1.0)
    {
        const double coefficient =
            data.passive_fiber_modulus * (fiber_stretch - 1.0) / fiber_stretch;
        Pe += coefficient * dyad(fiber, data.geometry.tangent);
    }

    const double shear = dot2(fiber, transverse);
    Pe += data.passive_shear_modulus * shear *
          (dyad(transverse, data.geometry.tangent) + dyad(fiber, data.geometry.normal));

    PP.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            for (unsigned int k = 0; k < NDIM; ++k)
            {
                PP(i, j) += J_active * Pe(i, k) * Fa_inverse(j, k);
            }
        }
    }
}

void
penalty_force_function(VectorValue<double>& F,
                       const TensorValue<double>& /*FF*/,
                       const libMesh::Point& x,
                       const libMesh::Point& X,
                       Elem* const /*elem*/,
                       const vector<const vector<double>*>& var_data,
                       const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                       const double time,
                       void* ctx)
{
    const auto& data = *static_cast<SwimmerData*>(ctx);
    const TargetState target = target_state(X, time, data);
    const vector<double>& velocity = *var_data[0];
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        F(d) = data.penalty_stiffness * (target.position(d) - x(d)) +
               data.penalty_damping * (target.velocity(d) - velocity[d]);
    }
}

ReferenceGeometry
build_reference_geometry(const Mesh& mesh, Pointer<Database> input_db)
{
    VectorValue<double> tangent;
    tangent(0) = input_db->getDoubleWithDefault("REFERENCE_TANGENT_X", 1.0);
    tangent(1) = input_db->getDoubleWithDefault("REFERENCE_TANGENT_Y", 0.0);
    const double tangent_norm = std::sqrt(dot2(tangent, tangent));
    if (tangent_norm <= 0.0) TBOX_ERROR("The reference tangent must be nonzero.\n");
    tangent /= tangent_norm;

    VectorValue<double> normal;
    normal(0) = -tangent(1);
    normal(1) = tangent(0);

    double s_min = std::numeric_limits<double>::max();
    double s_max = -std::numeric_limits<double>::max();
    double eta_min = std::numeric_limits<double>::max();
    double eta_max = -std::numeric_limits<double>::max();
    for (auto node_it = mesh.nodes_begin(); node_it != mesh.nodes_end(); ++node_it)
    {
        const Node& node = **node_it;
        const VectorValue<double> position(node(0), node(1));
        const double s = dot2(position, tangent);
        const double eta = dot2(position, normal);
        s_min = std::min(s_min, s);
        s_max = std::max(s_max, s);
        eta_min = std::min(eta_min, eta);
        eta_max = std::max(eta_max, eta);
    }
    IBTK_MPI::minReduction(&s_min, 1);
    IBTK_MPI::maxReduction(&s_max, 1);
    IBTK_MPI::minReduction(&eta_min, 1);
    IBTK_MPI::maxReduction(&eta_max, 1);

    ReferenceGeometry geometry;
    geometry.tangent = tangent;
    geometry.normal = normal;
    geometry.length = s_max - s_min;
    const double eta_center = 0.5 * (eta_min + eta_max);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        geometry.head(d) = s_min * tangent(d) + eta_center * normal(d);
    }
    return geometry;
}
} // namespace ModelData
using namespace ModelData;

static std::ofstream diagnostics_stream;

void
write_diagnostics(Mesh& mesh,
                  EquationSystems* equation_systems,
                  const std::string& coords_system_name,
                  const std::string& velocity_system_name,
                  const SwimmerData& swimmer_data,
                  const int iteration_num,
                  const double loop_time);

int
main(int argc, char* argv[])
{
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    const LibMeshInit& init = ibtk_init.getLibMeshInit();

    {
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();
#ifdef LIBMESH_HAVE_EXODUS_API
        const bool uses_exodus = dump_viz_data && !app_initializer->getExodusIIFilename().empty();
#else
        const bool uses_exodus = false;
        if (!app_initializer->getExodusIIFilename().empty())
        {
            plog << "WARNING: libMesh was compiled without Exodus support; FE output is disabled.\n";
        }
#endif
        const string exodus_filename = app_initializer->getExodusIIFilename();

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();
        const string restart_read_dirname = app_initializer->getRestartReadDirectory();
        const int restart_restore_num = app_initializer->getRestartRestoreNumber();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int postproc_data_dump_interval = app_initializer->getPostProcessingDataDumpInterval();
        const string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && postproc_data_dump_interval > 0 && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        Mesh mesh(init.comm(), NDIM);
        const string mesh_filename = input_db->getString("MESH_FILENAME");
        plog << "Reading volumetric FE mesh: " << mesh_filename << "\n";
        mesh.read(mesh_filename);
        mesh.prepare_for_use();
        if (mesh.mesh_dimension() != NDIM)
        {
            TBOX_ERROR("The external mesh must be a " << NDIM << "D volumetric mesh.\n");
        }

        const ReferenceGeometry reference_geometry = build_reference_geometry(mesh, input_db);
        SwimmerData swimmer_data(input_db, reference_geometry);
        void* const swimmer_data_ptr = static_cast<void*>(&swimmer_data);
        plog << "Swimmer stage: " << static_cast<int>(swimmer_data.stage)
             << ", reference length: " << swimmer_data.geometry.length << "\n";

        Pointer<INSHierarchyIntegrator> navier_stokes_integrator;
        const string solver_type = app_initializer->getComponentDatabase("Main")->getString("solver_type");
        if (solver_type == "STAGGERED")
        {
            navier_stokes_integrator = new INSStaggeredHierarchyIntegrator(
                "INSStaggeredHierarchyIntegrator",
                app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));
        }
        else if (solver_type == "COLLOCATED")
        {
            navier_stokes_integrator = new INSCollocatedHierarchyIntegrator(
                "INSCollocatedHierarchyIntegrator",
                app_initializer->getComponentDatabase("INSCollocatedHierarchyIntegrator"));
        }
        else
        {
            TBOX_ERROR("Unsupported solver type: " << solver_type << "\n");
        }

        Pointer<IBFEMethod> ib_method_ops =
            new IBFEMethod("IBFEMethod",
                           app_initializer->getComponentDatabase("IBFEMethod"),
                           &mesh,
                           app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"),
                           true,
                           restart_read_dirname,
                           restart_restore_num);
        Pointer<IBHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_method_ops,
                                              navier_stokes_integrator);
        Pointer<CartesianGridGeometry<NDIM> > grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM> > patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM> > error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM> > box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM> > load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM> > gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        ib_method_ops->initializeFEEquationSystems();
        EquationSystems* equation_systems = ib_method_ops->getFEDataManager()->getEquationSystems();
        const string coords_system_name = ib_method_ops->getCurrentCoordinatesSystemName();
        const string velocity_system_name = ib_method_ops->getVelocitySystemName();

        IBFEMethod::PK1StressFcnData PK1_stress_data(
            PK1_material_function, vector<SystemData>(), swimmer_data_ptr);
        PK1_stress_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(input_db->getString("PK1_QUAD_ORDER"));
        ib_method_ops->registerPK1StressFunction(PK1_stress_data);

        if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
        {
            vector<int> velocity_variables(NDIM);
            for (unsigned int d = 0; d < NDIM; ++d) velocity_variables[d] = d;
            vector<SystemData> system_data(1, SystemData(velocity_system_name, velocity_variables));
            IBFEMethod::LagBodyForceFcnData body_force_data(
                penalty_force_function, system_data, swimmer_data_ptr);
            ib_method_ops->registerLagBodyForceFunction(body_force_data);
        }

        if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
        {
            ib_method_ops->registerStressNormalizationPart();
        }

        if (input_db->keyExists("VelocityInitialConditions"))
        {
            Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
                "u_init", app_initializer->getComponentDatabase("VelocityInitialConditions"), grid_geometry);
            navier_stokes_integrator->registerVelocityInitialConditions(u_init);
        }
        if (input_db->keyExists("PressureInitialConditions"))
        {
            Pointer<CartGridFunction> p_init = new muParserCartGridFunction(
                "p_init", app_initializer->getComponentDatabase("PressureInitialConditions"), grid_geometry);
            navier_stokes_integrator->registerPressureInitialConditions(p_init);
        }

        const IntVector<NDIM>& periodic_shift = grid_geometry->getPeriodicShift();
        vector<RobinBcCoefStrategy<NDIM>*> velocity_bc_coefs(NDIM, nullptr);
        if (periodic_shift.min() <= 0)
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const string object_name = "u_bc_coefs_" + std::to_string(d);
                const string database_name = "VelocityBcCoefs_" + std::to_string(d);
                velocity_bc_coefs[d] = new muParserRobinBcCoefs(
                    object_name, app_initializer->getComponentDatabase(database_name), grid_geometry);
            }
            navier_stokes_integrator->registerPhysicalBoundaryConditions(velocity_bc_coefs);
        }

        Pointer<VisItDataWriter<NDIM> > visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit) time_integrator->registerVisItDataWriter(visit_data_writer);
        std::unique_ptr<ExodusII_IO> exodus_io = uses_exodus ? std::make_unique<ExodusII_IO>(mesh) : nullptr;
        if (uses_exodus)
        {
            exodus_io->append(RestartManager::getManager()->isFromRestart());
        }

        ib_method_ops->initializeFEData();
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        const bool from_restart = RestartManager::getManager()->isFromRestart();
        if (IBTK_MPI::getRank() == 0)
        {
            const string diagnostics_filename =
                input_db->getStringWithDefault("DIAGNOSTICS_FILENAME", "fish_diagnostics.csv");
            diagnostics_stream.open(
                diagnostics_filename, from_restart ? ios_base::out | ios_base::app : ios_base::out | ios_base::trunc);
            diagnostics_stream.precision(12);
            if (!from_restart)
            {
                diagnostics_stream
                    << "step,time,stage,J_min,J_max,lambda_active_min,lambda_active_max,"
                       "tracking_error_rms,tracking_error_max,penalty_force_x,penalty_force_y,"
                       "penalty_torque_z,penalty_power,tail_lateral,target_tail_lateral\n";
            }
        }

        app_initializer.setNull();
        plog << "Input database:\n";
        input_db->printClassData(plog);

        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        if (dump_viz_data)
        {
            if (uses_visit)
            {
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            }
            if (uses_exodus)
            {
                exodus_io->write_timestep(
                    exodus_filename, *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
            }
        }
        write_diagnostics(mesh,
                          equation_systems,
                          coords_system_name,
                          velocity_system_name,
                          swimmer_data,
                          iteration_num,
                          loop_time);

        const double loop_time_end = time_integrator->getEndTime();
        while (!IBTK::rel_equal_eps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();

            pout << "\nAt beginning of timestep " << iteration_num << ", time = " << loop_time << "\n";
            const double dt = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt);
            loop_time += dt;
            ++iteration_num;
            pout << "At end of timestep " << iteration_num << ", time = " << loop_time << "\n";

            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                if (uses_visit)
                {
                    time_integrator->setupPlotData();
                    visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                }
                if (uses_exodus)
                {
                    exodus_io->write_timestep(
                        exodus_filename, *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
                }
            }
            if (dump_restart_data && (iteration_num % restart_dump_interval == 0 || last_step))
            {
                RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
                ib_method_ops->writeFEDataToRestartFile(restart_dump_dirname, iteration_num);
            }
            if (dump_timer_data && (iteration_num % timer_dump_interval == 0 || last_step))
            {
                TimerManager::getManager()->print(plog);
            }
            if (dump_postproc_data &&
                (iteration_num % postproc_data_dump_interval == 0 || last_step))
            {
                write_diagnostics(mesh,
                                  equation_systems,
                                  coords_system_name,
                                  velocity_system_name,
                                  swimmer_data,
                                  iteration_num,
                                  loop_time);
            }
        }

        if (IBTK_MPI::getRank() == 0) diagnostics_stream.close();
        for (RobinBcCoefStrategy<NDIM>* bc_coef : velocity_bc_coefs) delete bc_coef;
    }
}

void
write_diagnostics(Mesh& mesh,
                  EquationSystems* equation_systems,
                  const std::string& coords_system_name,
                  const std::string& velocity_system_name,
                  const SwimmerData& swimmer_data,
                  const int iteration_num,
                  const double loop_time)
{
    const unsigned int dim = mesh.mesh_dimension();
    System& X_system = equation_systems->get_system(coords_system_name);
    System& U_system = equation_systems->get_system(velocity_system_name);
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    NumericVector<double>* U_ghost_vec = U_system.current_local_solution.get();
    copy_and_synch(*X_system.solution, *X_ghost_vec);
    copy_and_synch(*U_system.solution, *U_ghost_vec);

    const DofMap& dof_map = X_system.get_dof_map();
    vector<vector<unsigned int> > dof_indices(NDIM);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, dim, FIFTH);
    fe->attach_quadrature_rule(qrule.get());
    const vector<double>& JxW = fe->get_JxW();
    const vector<libMesh::Point>& reference_points = fe->get_xyz();
    const vector<vector<double> >& phi = fe->get_phi();
    const vector<vector<VectorValue<double> > >& dphi = fe->get_dphi();

    double volume = 0.0;
    double tracking_error_squared = 0.0;
    double tracking_error_max = 0.0;
    double penalty_force_x = 0.0;
    double penalty_force_y = 0.0;
    double penalty_torque = 0.0;
    double penalty_power = 0.0;
    double tail_weight = 0.0;
    double tail_lateral = 0.0;
    double target_tail_lateral = 0.0;
    double J_min = std::numeric_limits<double>::max();
    double J_max = -std::numeric_limits<double>::max();
    double lambda_min = std::numeric_limits<double>::max();
    double lambda_max = -std::numeric_limits<double>::max();

    boost::multi_array<double, 2> X_node;
    boost::multi_array<double, 2> U_node;
    VectorValue<double> x;
    VectorValue<double> velocity;
    TensorValue<double> FF;
    const libMesh::Point body_center = swimmer_data.geometry.head +
                                       0.5 * swimmer_data.geometry.length * swimmer_data.geometry.tangent;

    for (auto elem_it = mesh.active_local_elements_begin(); elem_it != mesh.active_local_elements_end(); ++elem_it)
    {
        Elem* const elem = *elem_it;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d) dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);
        get_values_for_interpolation(U_node, *U_ghost_vec, dof_indices);

        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            interpolate(x, qp, X_node, phi);
            interpolate(velocity, qp, U_node, phi);
            jacobian(FF, qp, X_node, dphi);
            const libMesh::Point& X = reference_points[qp];
            const double weight = JxW[qp];
            const double J = FF(0, 0) * FF(1, 1) - FF(0, 1) * FF(1, 0);
            const double lambda_active = active_stretch(X, loop_time, swimmer_data);

            volume += weight;
            J_min = std::min(J_min, J);
            J_max = std::max(J_max, J);
            lambda_min = std::min(lambda_min, lambda_active);
            lambda_max = std::max(lambda_max, lambda_active);

            TargetState target;
            target.position = X;
            target.velocity.zero();
            VectorValue<double> penalty_force;
            penalty_force.zero();
            if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
            {
                target = target_state(X, loop_time, swimmer_data);
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    penalty_force(d) =
                        swimmer_data.penalty_stiffness * (target.position(d) - x(d)) +
                        swimmer_data.penalty_damping * (target.velocity(d) - velocity(d));
                }
                const double error_x = target.position(0) - x(0);
                const double error_y = target.position(1) - x(1);
                const double error_squared = error_x * error_x + error_y * error_y;
                tracking_error_squared += error_squared * weight;
                tracking_error_max = std::max(tracking_error_max, std::sqrt(error_squared));
                penalty_force_x += penalty_force(0) * weight;
                penalty_force_y += penalty_force(1) * weight;
                penalty_torque += ((x(0) - body_center(0)) * penalty_force(1) -
                                   (x(1) - body_center(1)) * penalty_force(0)) *
                                  weight;
                penalty_power += dot2(penalty_force, target.velocity) * weight;
            }

            if (unit_coordinate(reference_s(X, swimmer_data), swimmer_data) >= 0.95)
            {
                const VectorValue<double> current_offset = x - swimmer_data.geometry.head;
                tail_weight += weight;
                tail_lateral += dot2(current_offset, swimmer_data.geometry.normal) * weight;
                if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
                {
                    const VectorValue<double> target_offset =
                        target.position - swimmer_data.geometry.head;
                    target_tail_lateral +=
                        dot2(target_offset, swimmer_data.geometry.normal) * weight;
                }
            }
        }
    }

    const std::array<double*, 8> sums = { &volume,
                                          &tracking_error_squared,
                                          &penalty_force_x,
                                          &penalty_force_y,
                                          &penalty_torque,
                                          &penalty_power,
                                          &tail_weight,
                                          &tail_lateral };
    for (double* value : sums) IBTK_MPI::sumReduction(value, 1);
    IBTK_MPI::sumReduction(&target_tail_lateral, 1);
    IBTK_MPI::maxReduction(&tracking_error_max, 1);
    IBTK_MPI::minReduction(&J_min, 1);
    IBTK_MPI::maxReduction(&J_max, 1);
    IBTK_MPI::minReduction(&lambda_min, 1);
    IBTK_MPI::maxReduction(&lambda_max, 1);

    const double tracking_error_rms =
        volume > 0.0 ? std::sqrt(tracking_error_squared / volume) : 0.0;
    if (tail_weight > 0.0)
    {
        tail_lateral /= tail_weight;
        target_tail_lateral /= tail_weight;
    }

    if (IBTK_MPI::getRank() == 0)
    {
        diagnostics_stream << iteration_num << "," << loop_time << ","
                           << static_cast<int>(swimmer_data.stage) << "," << J_min << "," << J_max << ","
                           << lambda_min << "," << lambda_max << "," << tracking_error_rms << ","
                           << tracking_error_max << "," << penalty_force_x << "," << penalty_force_y << ","
                           << penalty_torque << "," << penalty_power << "," << tail_lateral << ","
                           << target_tail_lateral << "\n";
        diagnostics_stream.flush();
    }
}

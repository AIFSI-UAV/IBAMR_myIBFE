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
    ACTIVE_STRAIN = 2,
    ACTIVE_STRAIN_TETHERED_TEST = 3
};

struct ReferenceGeometry
{
    VectorValue<double> tangent;
    VectorValue<double> normal;
    libMesh::Point head;
    double length = 0.0;
    double eta_min = 0.0;
    double eta_max = 0.0;
};

struct TargetState
{
    libMesh::Point position;
    VectorValue<double> velocity;
};

struct ActiveStretchState
{
    double curvature = 0.0;
    double raw = 1.0;
    double value = 1.0;
    bool clamped = false;
};

struct MaterialResponse
{
    TensorValue<double> total;
    TensorValue<double> matrix;
    TensorValue<double> fiber;
    TensorValue<double> shear;
    double J_elastic = 1.0;
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
    bool penalty_zero_net_force;
    bool penalty_zero_net_torque;
    VectorValue<double> penalty_projection_translation;
    double penalty_projection_rotation = 0.0;
    libMesh::Point penalty_projection_center;

    double passive_mu;
    double passive_lambda;
    double passive_fiber_modulus;
    double passive_shear_modulus;
    double stage1_passive_scale;

    double active_curvature_tail;
    double active_envelope_power;
    double active_stretch_min;
    double active_stretch_max;
    bool allow_active_stretch_clamp;
    double active_test_tether_fraction;
    double active_test_tether_stiffness;
    double active_test_tether_damping;

    bool run_material_self_checks;
    double material_self_check_tolerance;
    double diagnostic_station_half_width;

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
          penalty_zero_net_force(input_db->getBool("PENALTY_ZERO_NET_FORCE")),
          penalty_zero_net_torque(input_db->getBool("PENALTY_ZERO_NET_TORQUE")),
          passive_mu(input_db->getDouble("PASSIVE_MU")),
          passive_lambda(input_db->getDouble("PASSIVE_LAMBDA")),
          passive_fiber_modulus(input_db->getDouble("PASSIVE_FIBER_MODULUS")),
          passive_shear_modulus(input_db->getDouble("PASSIVE_SHEAR_MODULUS")),
          stage1_passive_scale(input_db->getDouble("STAGE1_PASSIVE_SCALE")),
          active_curvature_tail(input_db->getDouble("ACTIVE_CURVATURE_TAIL_TIMES_L") /
                                reference_geometry.length),
          active_envelope_power(input_db->getDouble("ACTIVE_ENVELOPE_POWER")),
          active_stretch_min(input_db->getDouble("ACTIVE_STRETCH_MIN")),
          active_stretch_max(input_db->getDouble("ACTIVE_STRETCH_MAX")),
          allow_active_stretch_clamp(input_db->getBool("ALLOW_ACTIVE_STRETCH_CLAMP")),
          active_test_tether_fraction(input_db->getDouble("ACTIVE_TEST_TETHER_FRACTION")),
          active_test_tether_stiffness(input_db->getDouble("ACTIVE_TEST_TETHER_STIFFNESS")),
          active_test_tether_damping(input_db->getDouble("ACTIVE_TEST_TETHER_DAMPING")),
          run_material_self_checks(input_db->getBool("RUN_MATERIAL_SELF_CHECKS")),
          material_self_check_tolerance(input_db->getDouble("MATERIAL_SELF_CHECK_TOLERANCE")),
          diagnostic_station_half_width(input_db->getDouble("DIAGNOSTIC_STATION_HALF_WIDTH"))
    {
        const int stage_number = static_cast<int>(stage);
        if (stage_number < 1 || stage_number > 3)
        {
            TBOX_ERROR("SWIMMER_STAGE must be 1 (penalty), 2 (active strain), or 3 "
                       "(tethered active-strain test).\n");
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
        if (stage1_passive_scale < 0.0)
        {
            TBOX_ERROR("STAGE1_PASSIVE_SCALE must be nonnegative.\n");
        }
        if (active_envelope_power < 0.0)
        {
            TBOX_ERROR("ACTIVE_ENVELOPE_POWER must be nonnegative.\n");
        }
        if (!(0.0 < active_stretch_min && active_stretch_min < 1.0 && 1.0 < active_stretch_max))
        {
            TBOX_ERROR("Require 0 < ACTIVE_STRETCH_MIN < 1 < ACTIVE_STRETCH_MAX.\n");
        }
        if (!(0.0 < active_test_tether_fraction && active_test_tether_fraction <= 1.0))
        {
            TBOX_ERROR("ACTIVE_TEST_TETHER_FRACTION must be in (0,1].\n");
        }
        if (active_test_tether_stiffness < 0.0 || active_test_tether_damping < 0.0)
        {
            TBOX_ERROR("Active-test tether coefficients must be nonnegative.\n");
        }
        if (stage == SwimmerStage::ACTIVE_STRAIN_TETHERED_TEST &&
            active_test_tether_stiffness <= 0.0)
        {
            TBOX_ERROR("Stage 3 requires ACTIVE_TEST_TETHER_STIFFNESS > 0.\n");
        }
        if (material_self_check_tolerance <= 0.0)
        {
            TBOX_ERROR("MATERIAL_SELF_CHECK_TOLERANCE must be positive.\n");
        }
        if (!(0.0 < diagnostic_station_half_width && diagnostic_station_half_width < 0.1))
        {
            TBOX_ERROR("DIAGNOSTIC_STATION_HALF_WIDTH must be in (0,0.1).\n");
        }
        penalty_projection_translation.zero();
        penalty_projection_center = geometry.head + 0.5 * geometry.length * geometry.tangent;
    }
};

struct PenaltyProjectionContext
{
    Mesh* mesh = nullptr;
    EquationSystems* equation_systems = nullptr;
    std::string coords_system_name;
    std::string velocity_system_name;
    SwimmerData* swimmer_data = nullptr;
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

bool
uses_active_strain(const SwimmerData& data)
{
    return data.stage == SwimmerStage::ACTIVE_STRAIN ||
           data.stage == SwimmerStage::ACTIVE_STRAIN_TETHERED_TEST;
}

ActiveStretchState
active_stretch_state(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    ActiveStretchState state;
    if (!uses_active_strain(data)) return state;
    state.curvature = active_curvature(X, time, data);
    state.raw = 1.0 - reference_eta(X, data) * state.curvature;
    state.value = std::clamp(state.raw, data.active_stretch_min, data.active_stretch_max);
    state.clamped = !IBTK::rel_equal_eps(state.raw, state.value);
    return state;
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

double
frobenius_norm(const TensorValue<double>& tensor)
{
    double norm_squared = 0.0;
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j) norm_squared += tensor(i, j) * tensor(i, j);
    }
    return std::sqrt(norm_squared);
}

TensorValue<double>
multiply2(const TensorValue<double>& left, const TensorValue<double>& right)
{
    TensorValue<double> result;
    result.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            for (unsigned int k = 0; k < NDIM; ++k) result(i, j) += left(i, k) * right(k, j);
        }
    }
    result(2, 2) = 1.0;
    return result;
}

TensorValue<double>
pull_back_active_stress(const TensorValue<double>& elastic_stress,
                        const TensorValue<double>& Fa_inverse,
                        const double J_active)
{
    TensorValue<double> result;
    result.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            for (unsigned int k = 0; k < NDIM; ++k)
            {
                result(i, j) += J_active * elastic_stress(i, k) * Fa_inverse(j, k);
            }
        }
    }
    return result;
}

MaterialResponse
evaluate_material(const TensorValue<double>& FF,
                  const double lambda_active,
                  const SwimmerData& data,
                  const libMesh::Point* X,
                  const double time)
{
    const TensorValue<double> tt = dyad(data.geometry.tangent, data.geometry.tangent);
    const TensorValue<double> nn = dyad(data.geometry.normal, data.geometry.normal);
    TensorValue<double> Fa_inverse = (1.0 / lambda_active) * tt + nn;
    Fa_inverse(2, 2) = 1.0;
    const TensorValue<double> Fe = multiply2(FF, Fa_inverse);

    MaterialResponse response;
    response.J_elastic = Fe(0, 0) * Fe(1, 1) - Fe(0, 1) * Fe(1, 0);
    if (!(response.J_elastic > 0.0) || !std::isfinite(response.J_elastic))
    {
        if (X)
        {
            TBOX_ERROR("Nonpositive elastic Jacobian in active-strain material at time "
                       << time << ", X=(" << (*X)(0) << "," << (*X)(1)
                       << "), Je=" << response.J_elastic << "\n");
        }
        TBOX_ERROR("Nonpositive elastic Jacobian in active-strain material self-check: Je="
                   << response.J_elastic << "\n");
    }

    const double material_scale =
        data.stage == SwimmerStage::PENALTY_PRESCRIBED ? data.stage1_passive_scale : 1.0;
    const TensorValue<double> Fe_inverse_transpose = tensor_inverse_transpose(Fe, NDIM);
    TensorValue<double> Pe_matrix;
    Pe_matrix.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            Pe_matrix(i, j) =
                material_scale *
                (data.passive_mu * (Fe(i, j) - Fe_inverse_transpose(i, j)) +
                 data.passive_lambda * std::log(response.J_elastic) * Fe_inverse_transpose(i, j));
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

    TensorValue<double> Pe_fiber;
    Pe_fiber.zero();
    const double fiber_stretch = std::sqrt(dot2(fiber, fiber));
    if (fiber_stretch > 1.0)
    {
        const double coefficient = material_scale * data.passive_fiber_modulus *
                                   (fiber_stretch - 1.0) / fiber_stretch;
        Pe_fiber = coefficient * dyad(fiber, data.geometry.tangent);
    }

    const double shear_measure = dot2(fiber, transverse);
    const TensorValue<double> Pe_shear =
        material_scale * data.passive_shear_modulus * shear_measure *
        (dyad(transverse, data.geometry.tangent) + dyad(fiber, data.geometry.normal));

    response.matrix = pull_back_active_stress(Pe_matrix, Fa_inverse, lambda_active);
    response.fiber = pull_back_active_stress(Pe_fiber, Fa_inverse, lambda_active);
    response.shear = pull_back_active_stress(Pe_shear, Fa_inverse, lambda_active);
    response.total = response.matrix + response.fiber + response.shear;
    return response;
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
    const ActiveStretchState stretch = active_stretch_state(X, time, data);
    PP = evaluate_material(FF, stretch.value, data, &X, time).total;
}

void
raw_penalty_force(VectorValue<double>& F,
                  const libMesh::Point& x,
                  const libMesh::Point& X,
                  const VectorValue<double>& velocity,
                  const double time,
                  const SwimmerData& data)
{
    const TargetState target = target_state(X, time, data);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        F(d) = data.penalty_stiffness * (target.position(d) - x(d)) +
               data.penalty_damping * (target.velocity(d) - velocity(d));
    }
}

double
smooth_head_weight(const double xi, const double tether_fraction)
{
    if (xi >= tether_fraction) return 0.0;
    const double coordinate = std::clamp(xi / tether_fraction, 0.0, 1.0);
    return 0.5 * (1.0 + std::cos(M_PI * coordinate));
}

void
evaluate_lag_body_force(VectorValue<double>& F,
                        const libMesh::Point& x,
                        const libMesh::Point& X,
                        const VectorValue<double>& velocity,
                        const double time,
                        const SwimmerData& data)
{
    F.zero();
    if (data.stage == SwimmerStage::PENALTY_PRESCRIBED)
    {
        raw_penalty_force(F, x, X, velocity, time, data);
        if (data.penalty_zero_net_force) F -= data.penalty_projection_translation;
        if (data.penalty_zero_net_torque)
        {
            const VectorValue<double> radius = x - data.penalty_projection_center;
            F(0) += data.penalty_projection_rotation * radius(1);
            F(1) -= data.penalty_projection_rotation * radius(0);
        }
    }
    else if (data.stage == SwimmerStage::ACTIVE_STRAIN_TETHERED_TEST)
    {
        const double weight =
            smooth_head_weight(unit_coordinate(reference_s(X, data), data), data.active_test_tether_fraction);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            F(d) = weight * (data.active_test_tether_stiffness * (X(d) - x(d)) -
                             data.active_test_tether_damping * velocity(d));
        }
    }
}

void
lag_body_force_function(VectorValue<double>& F,
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
    VectorValue<double> velocity;
    for (unsigned int d = 0; d < NDIM; ++d) velocity(d) = (*var_data[0])[d];
    evaluate_lag_body_force(F, x, X, velocity, time, data);
}

ReferenceGeometry
build_reference_geometry(const Mesh& mesh, Pointer<Database> input_db)
{
    VectorValue<double> axis;
    axis(0) = input_db->getDoubleWithDefault("REFERENCE_AXIS_X", 1.0);
    axis(1) = input_db->getDoubleWithDefault("REFERENCE_AXIS_Y", 0.0);
    const double axis_norm = std::sqrt(dot2(axis, axis));
    if (axis_norm <= 0.0) TBOX_ERROR("The reference axis must be nonzero.\n");
    axis /= axis_norm;

    double projection_min = std::numeric_limits<double>::max();
    double projection_max = -std::numeric_limits<double>::max();
    for (auto node_it = mesh.nodes_begin(); node_it != mesh.nodes_end(); ++node_it)
    {
        const Node& node = **node_it;
        const VectorValue<double> position(node(0), node(1));
        const double projection = dot2(position, axis);
        projection_min = std::min(projection_min, projection);
        projection_max = std::max(projection_max, projection);
    }
    IBTK_MPI::minReduction(&projection_min, 1);
    IBTK_MPI::maxReduction(&projection_max, 1);

    const std::string head_end = input_db->getString("REFERENCE_HEAD_END");
    VectorValue<double> tangent;
    double head_projection = 0.0;
    if (head_end == "MAX_PROJECTION")
    {
        tangent = -axis;
        head_projection = projection_max;
    }
    else if (head_end == "MIN_PROJECTION")
    {
        tangent = axis;
        head_projection = projection_min;
    }
    else
    {
        TBOX_ERROR("REFERENCE_HEAD_END must be MAX_PROJECTION or MIN_PROJECTION.\n");
    }

    VectorValue<double> normal;
    normal(0) = -tangent(1);
    normal(1) = tangent(0);
    double eta_min = std::numeric_limits<double>::max();
    double eta_max = -std::numeric_limits<double>::max();
    for (auto node_it = mesh.nodes_begin(); node_it != mesh.nodes_end(); ++node_it)
    {
        const Node& node = **node_it;
        const VectorValue<double> position(node(0), node(1));
        const double eta = dot2(position, normal);
        eta_min = std::min(eta_min, eta);
        eta_max = std::max(eta_max, eta);
    }
    IBTK_MPI::minReduction(&eta_min, 1);
    IBTK_MPI::maxReduction(&eta_max, 1);

    ReferenceGeometry geometry;
    geometry.tangent = tangent;
    geometry.normal = normal;
    geometry.length = projection_max - projection_min;
    geometry.eta_min = eta_min;
    geometry.eta_max = eta_max;
    const double eta_center = 0.5 * (eta_min + eta_max);
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        geometry.head(d) = head_projection * axis(d) + eta_center * normal(d);
    }
    return geometry;
}

void
update_penalty_projection(const double current_time,
                          const double new_time,
                          const int /*num_cycles*/,
                          void* ctx)
{
    auto& projection = *static_cast<PenaltyProjectionContext*>(ctx);
    SwimmerData& data = *projection.swimmer_data;
    data.penalty_projection_translation.zero();
    data.penalty_projection_rotation = 0.0;
    if (data.stage != SwimmerStage::PENALTY_PRESCRIBED ||
        (!data.penalty_zero_net_force && !data.penalty_zero_net_torque))
    {
        return;
    }

    Mesh& mesh = *projection.mesh;
    EquationSystems& equation_systems = *projection.equation_systems;
    System& X_system = equation_systems.get_system(projection.coords_system_name);
    System& U_system = equation_systems.get_system(projection.velocity_system_name);
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    NumericVector<double>* U_ghost_vec = U_system.current_local_solution.get();
    copy_and_synch(*X_system.solution, *X_ghost_vec);
    copy_and_synch(*U_system.solution, *U_ghost_vec);

    const DofMap& dof_map = X_system.get_dof_map();
    vector<vector<unsigned int> > dof_indices(NDIM);
    std::unique_ptr<FEBase> fe(FEBase::build(mesh.mesh_dimension(), dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, mesh.mesh_dimension(), FIFTH);
    fe->attach_quadrature_rule(qrule.get());
    const vector<double>& JxW = fe->get_JxW();
    const vector<libMesh::Point>& reference_points = fe->get_xyz();
    const vector<vector<double> >& phi = fe->get_phi();

    boost::multi_array<double, 2> X_node;
    boost::multi_array<double, 2> U_node;
    VectorValue<double> x;
    VectorValue<double> velocity;
    VectorValue<double> raw_force;
    const double projection_time = 0.5 * (current_time + new_time);

    double volume = 0.0;
    double center_integral[NDIM] = { 0.0, 0.0 };
    double force_integral[NDIM] = { 0.0, 0.0 };
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
            raw_penalty_force(raw_force, x, reference_points[qp], velocity, projection_time, data);
            const double weight = JxW[qp];
            volume += weight;
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                center_integral[d] += x(d) * weight;
                force_integral[d] += raw_force(d) * weight;
            }
        }
    }
    IBTK_MPI::sumReduction(&volume, 1);
    IBTK_MPI::sumReduction(center_integral, NDIM);
    IBTK_MPI::sumReduction(force_integral, NDIM);
    if (volume <= 0.0) TBOX_ERROR("Cannot project penalty force on a zero-volume mesh.\n");

    for (unsigned int d = 0; d < NDIM; ++d)
    {
        data.penalty_projection_center(d) = center_integral[d] / volume;
        if (data.penalty_zero_net_force)
        {
            data.penalty_projection_translation(d) = force_integral[d] / volume;
        }
    }

    if (!data.penalty_zero_net_torque) return;
    double torque = 0.0;
    double polar_moment = 0.0;
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
            raw_penalty_force(raw_force, x, reference_points[qp], velocity, projection_time, data);
            if (data.penalty_zero_net_force) raw_force -= data.penalty_projection_translation;
            const VectorValue<double> radius = x - data.penalty_projection_center;
            const double weight = JxW[qp];
            torque += (radius(0) * raw_force(1) - radius(1) * raw_force(0)) * weight;
            polar_moment += dot2(radius, radius) * weight;
        }
    }
    IBTK_MPI::sumReduction(&torque, 1);
    IBTK_MPI::sumReduction(&polar_moment, 1);
    if (polar_moment <= 0.0) TBOX_ERROR("Cannot project penalty torque with zero polar moment.\n");
    data.penalty_projection_rotation = torque / polar_moment;
}

void
postprocess_penalty_projection(const double /*current_time*/,
                               const double new_time,
                               const bool /*skip_synchronize_new_state_data*/,
                               const int num_cycles,
                               void* ctx)
{
    update_penalty_projection(new_time, new_time, num_cycles, ctx);
}

void
run_material_self_checks(const SwimmerData& data)
{
    if (!data.run_material_self_checks) return;

    TensorValue<double> identity;
    identity.zero();
    for (unsigned int d = 0; d < 3; ++d) identity(d, d) = 1.0;
    const MaterialResponse passive = evaluate_material(identity, 1.0, data, nullptr, 0.0);
    const double passive_residual = frobenius_norm(passive.total);

    const double check_time = data.ramp_time + 0.25 / data.frequency;
    const double eta = 0.5 * (data.geometry.eta_min + data.geometry.eta_max) +
                       0.45 * (data.geometry.eta_max - data.geometry.eta_min);
    const libMesh::Point X = data.geometry.head + 0.8 * data.geometry.length * data.geometry.tangent +
                             eta * data.geometry.normal;
    ActiveStretchState stretch = active_stretch_state(X, check_time, data);
    if (!uses_active_strain(data))
    {
        stretch.curvature = data.active_curvature_tail * std::pow(0.8, data.active_envelope_power) *
                            std::sin(wave_number(data) * 0.8 * data.geometry.length -
                                     angular_frequency(data) * check_time + data.phase0);
        stretch.raw = 1.0 - eta * stretch.curvature;
        stretch.value = std::clamp(stretch.raw, data.active_stretch_min, data.active_stretch_max);
    }
    TensorValue<double> Fa = stretch.value * dyad(data.geometry.tangent, data.geometry.tangent) +
                             dyad(data.geometry.normal, data.geometry.normal);
    Fa(2, 2) = 1.0;
    const MaterialResponse active_stress_free =
        evaluate_material(Fa, stretch.value, data, nullptr, check_time);
    const double active_residual = frobenius_norm(active_stress_free.total);

    pout << "Material self-check: ||P(F=I,Fa=I)|| = " << passive_residual
         << ", ||P(F=Fa)|| = " << active_residual << "\n";
    if (!std::isfinite(passive_residual) || !std::isfinite(active_residual) ||
        passive_residual > data.material_self_check_tolerance ||
        active_residual > data.material_self_check_tolerance)
    {
        TBOX_ERROR("Active-strain material self-check failed. Tolerance = "
                   << data.material_self_check_tolerance << "\n");
    }
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
        const libMesh::Point reference_tail =
            swimmer_data.geometry.head + swimmer_data.geometry.length * swimmer_data.geometry.tangent;
        plog << "Reference head = (" << swimmer_data.geometry.head(0) << ","
             << swimmer_data.geometry.head(1) << "), tail = (" << reference_tail(0) << ","
             << reference_tail(1) << "), length = " << swimmer_data.geometry.length << "\n";
        pout << "Reference convention: s=0 is head at (" << swimmer_data.geometry.head(0) << ","
             << swimmer_data.geometry.head(1) << "), s=L is tail at (" << reference_tail(0) << ","
             << reference_tail(1) << ").\n";
        if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
        {
            pout << "=== Stage 1: prescribed kinematics via penalty body force ===\n";
            pout << "Stage-1 passive material scale = " << swimmer_data.stage1_passive_scale << ".\n";
            if (!swimmer_data.penalty_zero_net_force || !swimmer_data.penalty_zero_net_torque)
            {
                pout << "This is a tethered wake-validation case. Unprojected penalty force/torque makes "
                        "free-swimming speed and efficiency nonphysical.\n";
            }
            else
            {
                pout << "Penalty rigid-force and rigid-torque modes are projected once per time step. "
                        "Swimming speed is an approximate prescribed-gait result; penalty efficiency remains "
                        "nonphysical.\n";
            }
        }
        else if (swimmer_data.stage == SwimmerStage::ACTIVE_STRAIN)
        {
            pout << "=== Stage 2: free active-strain FSI ===\n";
        }
        else
        {
            pout << "=== Stage 3: head-tethered active-strain structural/FSI test ===\n";
        }

        const double eta_abs_max =
            std::max(std::abs(swimmer_data.geometry.eta_min), std::abs(swimmer_data.geometry.eta_max));
        const double raw_stretch_lower = 1.0 - eta_abs_max * std::abs(swimmer_data.active_curvature_tail);
        const double raw_stretch_upper = 1.0 + eta_abs_max * std::abs(swimmer_data.active_curvature_tail);
        pout << "Worst-case unclamped active stretch bound = [" << raw_stretch_lower << ","
             << raw_stretch_upper << "].\n";
        if (raw_stretch_lower <= 0.0)
        {
            TBOX_ERROR("ACTIVE_CURVATURE_TAIL_TIMES_L permits a nonpositive raw active stretch.\n");
        }
        if (uses_active_strain(swimmer_data) && !swimmer_data.allow_active_stretch_clamp &&
            (raw_stretch_lower < swimmer_data.active_stretch_min ||
             raw_stretch_upper > swimmer_data.active_stretch_max))
        {
            TBOX_ERROR("The configured active curvature reaches the active-stretch clamp. Reduce "
                       "ACTIVE_CURVATURE_TAIL_TIMES_L or set ALLOW_ACTIVE_STRETCH_CLAMP=TRUE "
                       "for an intentional clipped test.\n");
        }
        run_material_self_checks(swimmer_data);

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
        PenaltyProjectionContext penalty_projection_context{
            &mesh, equation_systems, coords_system_name, velocity_system_name, &swimmer_data
        };

        IBFEMethod::PK1StressFcnData PK1_stress_data(
            PK1_material_function, vector<SystemData>(), swimmer_data_ptr);
        PK1_stress_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(input_db->getString("PK1_QUAD_ORDER"));
        ib_method_ops->registerPK1StressFunction(PK1_stress_data);

        if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED ||
            swimmer_data.stage == SwimmerStage::ACTIVE_STRAIN_TETHERED_TEST)
        {
            vector<int> velocity_variables(NDIM);
            for (unsigned int d = 0; d < NDIM; ++d) velocity_variables[d] = d;
            vector<SystemData> system_data(1, SystemData(velocity_system_name, velocity_variables));
            IBFEMethod::LagBodyForceFcnData body_force_data(
                lag_body_force_function, system_data, swimmer_data_ptr);
            ib_method_ops->registerLagBodyForceFunction(body_force_data);
        }
        if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED &&
            (swimmer_data.penalty_zero_net_force || swimmer_data.penalty_zero_net_torque))
        {
            time_integrator->registerPreprocessIntegrateHierarchyCallback(
                update_penalty_projection, &penalty_projection_context);
            time_integrator->registerPostprocessIntegrateHierarchyCallback(
                postprocess_penalty_projection, &penalty_projection_context);
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
        if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED &&
            (swimmer_data.penalty_zero_net_force || swimmer_data.penalty_zero_net_torque))
        {
            const double initial_time = time_integrator->getIntegratorTime();
            update_penalty_projection(initial_time, initial_time, 1, &penalty_projection_context);
        }

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
                    << "step,time,stage,J_total_min,J_total_max,J_elastic_min,J_elastic_max,"
                       "lambda_raw_min,lambda_raw_max,lambda_active_min,lambda_active_max,clamp_fraction,"
                       "P_matrix_mean,P_fiber_mean,P_shear_mean,"
                       "tracking_error_rms,tracking_error_max,tracking_error_rms_over_L,"
                       "tracking_error_max_over_L,body_force_x,body_force_y,body_torque_z,body_power,"
                       "tail_root_lateral,tail_tip_lateral,tail_lateral_over_L,tail_pitch,"
                       "target_tail_root_lateral,target_tail_tip_lateral,target_tail_pitch,"
                       "actual_curvature_mid,actual_curvature_tail,target_curvature_mid,"
                       "target_curvature_tail\n";
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
    static const std::array<double, 8> station_xi = { 0.02, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98 };
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
    double clamp_volume = 0.0;
    double tracking_error_squared = 0.0;
    double tracking_error_max = 0.0;
    double body_force_x = 0.0;
    double body_force_y = 0.0;
    double body_torque = 0.0;
    double body_power = 0.0;
    double P_matrix_integral = 0.0;
    double P_fiber_integral = 0.0;
    double P_shear_integral = 0.0;
    double J_total_min = std::numeric_limits<double>::max();
    double J_total_max = -std::numeric_limits<double>::max();
    double J_elastic_min = std::numeric_limits<double>::max();
    double J_elastic_max = -std::numeric_limits<double>::max();
    double lambda_raw_min = std::numeric_limits<double>::max();
    double lambda_raw_max = -std::numeric_limits<double>::max();
    double lambda_active_min = std::numeric_limits<double>::max();
    double lambda_active_max = -std::numeric_limits<double>::max();
    std::array<double, station_xi.size()> station_weight{};
    std::array<double, station_xi.size()> station_x{};
    std::array<double, station_xi.size()> station_y{};
    std::array<double, station_xi.size()> target_station_x{};
    std::array<double, station_xi.size()> target_station_y{};

    boost::multi_array<double, 2> X_node;
    boost::multi_array<double, 2> U_node;
    VectorValue<double> x;
    VectorValue<double> velocity;
    TensorValue<double> FF;
    libMesh::Point torque_center = swimmer_data.penalty_projection_center;
    if (swimmer_data.stage != SwimmerStage::PENALTY_PRESCRIBED)
    {
        torque_center = swimmer_data.geometry.head;
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            torque_center(d) += 0.5 * swimmer_data.geometry.length * swimmer_data.geometry.tangent(d);
        }
    }

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
            const double xi = unit_coordinate(reference_s(X, swimmer_data), swimmer_data);
            const double J_total = FF(0, 0) * FF(1, 1) - FF(0, 1) * FF(1, 0);
            const ActiveStretchState stretch = active_stretch_state(X, loop_time, swimmer_data);
            const MaterialResponse material =
                evaluate_material(FF, stretch.value, swimmer_data, &X, loop_time);

            volume += weight;
            if (stretch.clamped) clamp_volume += weight;
            J_total_min = std::min(J_total_min, J_total);
            J_total_max = std::max(J_total_max, J_total);
            J_elastic_min = std::min(J_elastic_min, material.J_elastic);
            J_elastic_max = std::max(J_elastic_max, material.J_elastic);
            lambda_raw_min = std::min(lambda_raw_min, stretch.raw);
            lambda_raw_max = std::max(lambda_raw_max, stretch.raw);
            lambda_active_min = std::min(lambda_active_min, stretch.value);
            lambda_active_max = std::max(lambda_active_max, stretch.value);
            P_matrix_integral += frobenius_norm(material.matrix) * weight;
            P_fiber_integral += frobenius_norm(material.fiber) * weight;
            P_shear_integral += frobenius_norm(material.shear) * weight;

            TargetState target;
            target.position = X;
            target.velocity.zero();
            VectorValue<double> body_force;
            evaluate_lag_body_force(body_force, x, X, velocity, loop_time, swimmer_data);
            if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
            {
                target = target_state(X, loop_time, swimmer_data);
                const double error_x = target.position(0) - x(0);
                const double error_y = target.position(1) - x(1);
                const double error_squared = error_x * error_x + error_y * error_y;
                tracking_error_squared += error_squared * weight;
                tracking_error_max = std::max(tracking_error_max, std::sqrt(error_squared));
            }

            body_force_x += body_force(0) * weight;
            body_force_y += body_force(1) * weight;
            body_torque += ((x(0) - torque_center(0)) * body_force(1) -
                            (x(1) - torque_center(1)) * body_force(0)) *
                           weight;
            body_power += dot2(body_force, velocity) * weight;

            for (std::size_t station = 0; station < station_xi.size(); ++station)
            {
                const double distance = std::abs(xi - station_xi[station]);
                if (distance >= swimmer_data.diagnostic_station_half_width) continue;
                const double station_kernel =
                    1.0 - distance / swimmer_data.diagnostic_station_half_width;
                const double station_quadrature_weight = station_kernel * weight;
                station_weight[station] += station_quadrature_weight;
                station_x[station] += x(0) * station_quadrature_weight;
                station_y[station] += x(1) * station_quadrature_weight;
                if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
                {
                    target_station_x[station] += target.position(0) * station_quadrature_weight;
                    target_station_y[station] += target.position(1) * station_quadrature_weight;
                }
            }
        }
    }

    const std::array<double*, 10> sums = { &volume,
                                           &clamp_volume,
                                           &tracking_error_squared,
                                           &body_force_x,
                                           &body_force_y,
                                           &body_torque,
                                           &body_power,
                                           &P_matrix_integral,
                                           &P_fiber_integral,
                                           &P_shear_integral };
    for (double* value : sums) IBTK_MPI::sumReduction(value, 1);
    IBTK_MPI::sumReduction(station_weight.data(), station_weight.size());
    IBTK_MPI::sumReduction(station_x.data(), station_x.size());
    IBTK_MPI::sumReduction(station_y.data(), station_y.size());
    IBTK_MPI::sumReduction(target_station_x.data(), target_station_x.size());
    IBTK_MPI::sumReduction(target_station_y.data(), target_station_y.size());
    IBTK_MPI::maxReduction(&tracking_error_max, 1);
    IBTK_MPI::minReduction(&J_total_min, 1);
    IBTK_MPI::maxReduction(&J_total_max, 1);
    IBTK_MPI::minReduction(&J_elastic_min, 1);
    IBTK_MPI::maxReduction(&J_elastic_max, 1);
    IBTK_MPI::minReduction(&lambda_raw_min, 1);
    IBTK_MPI::maxReduction(&lambda_raw_max, 1);
    IBTK_MPI::minReduction(&lambda_active_min, 1);
    IBTK_MPI::maxReduction(&lambda_active_max, 1);

    const double tracking_error_rms =
        volume > 0.0 ? std::sqrt(tracking_error_squared / volume) : 0.0;
    const double clamp_fraction = volume > 0.0 ? clamp_volume / volume : 0.0;
    const double P_matrix_mean = volume > 0.0 ? P_matrix_integral / volume : 0.0;
    const double P_fiber_mean = volume > 0.0 ? P_fiber_integral / volume : 0.0;
    const double P_shear_mean = volume > 0.0 ? P_shear_integral / volume : 0.0;

    std::array<libMesh::Point, station_xi.size()> stations;
    std::array<libMesh::Point, station_xi.size()> target_stations;
    for (std::size_t station = 0; station < station_xi.size(); ++station)
    {
        if (station_weight[station] <= 0.0)
        {
            TBOX_ERROR("No diagnostic quadrature points found near xi=" << station_xi[station]
                                                                        << ". Increase "
                                                                           "DIAGNOSTIC_STATION_HALF_WIDTH.\n");
        }
        stations[station](0) = station_x[station] / station_weight[station];
        stations[station](1) = station_y[station] / station_weight[station];
        if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
        {
            target_stations[station](0) = target_station_x[station] / station_weight[station];
            target_stations[station](1) = target_station_y[station] / station_weight[station];
        }
    }

    const auto lateral = [&](const libMesh::Point& point) {
        return dot2(point - swimmer_data.geometry.head, swimmer_data.geometry.normal);
    };
    const auto pitch = [&](const libMesh::Point& root, const libMesh::Point& tip) {
        const VectorValue<double> direction = tip - root;
        return std::atan2(dot2(direction, swimmer_data.geometry.normal),
                          dot2(direction, swimmer_data.geometry.tangent));
    };
    const auto curvature = [](const libMesh::Point& a,
                              const libMesh::Point& b,
                              const libMesh::Point& c) {
        const double ab = (b - a).norm();
        const double bc = (c - b).norm();
        const double ac = (c - a).norm();
        const double denominator = ab * bc * ac;
        if (denominator <= std::numeric_limits<double>::epsilon()) return 0.0;
        const double cross = (b(0) - a(0)) * (c(1) - a(1)) -
                             (b(1) - a(1)) * (c(0) - a(0));
        return 2.0 * cross / denominator;
    };

    const double tail_root_lateral = lateral(stations[5]);
    const double tail_tip_lateral = lateral(stations[7]);
    const double tail_lateral_over_L =
        (tail_tip_lateral - lateral(stations[0])) / swimmer_data.geometry.length;
    const double tail_pitch = pitch(stations[5], stations[7]);
    const double actual_curvature_mid = curvature(stations[1], stations[2], stations[3]);
    const double actual_curvature_tail = curvature(stations[4], stations[5], stations[6]);

    double target_tail_root_lateral = std::numeric_limits<double>::quiet_NaN();
    double target_tail_tip_lateral = std::numeric_limits<double>::quiet_NaN();
    double target_tail_pitch = std::numeric_limits<double>::quiet_NaN();
    double target_curvature_mid = 0.0;
    double target_curvature_tail = 0.0;
    if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
    {
        target_tail_root_lateral = lateral(target_stations[5]);
        target_tail_tip_lateral = lateral(target_stations[7]);
        target_tail_pitch = pitch(target_stations[5], target_stations[7]);
        target_curvature_mid = curvature(target_stations[1], target_stations[2], target_stations[3]);
        target_curvature_tail = curvature(target_stations[4], target_stations[5], target_stations[6]);
    }
    else
    {
        const libMesh::Point X_mid =
            swimmer_data.geometry.head + station_xi[2] * swimmer_data.geometry.length *
                                             swimmer_data.geometry.tangent;
        const libMesh::Point X_tail =
            swimmer_data.geometry.head + station_xi[5] * swimmer_data.geometry.length *
                                             swimmer_data.geometry.tangent;
        target_curvature_mid = active_curvature(X_mid, loop_time, swimmer_data);
        target_curvature_tail = active_curvature(X_tail, loop_time, swimmer_data);
    }

    if (IBTK_MPI::getRank() == 0)
    {
        diagnostics_stream << iteration_num << "," << loop_time << ","
                           << static_cast<int>(swimmer_data.stage) << "," << J_total_min << ","
                           << J_total_max << "," << J_elastic_min << "," << J_elastic_max << ","
                           << lambda_raw_min << "," << lambda_raw_max << "," << lambda_active_min << ","
                           << lambda_active_max << "," << clamp_fraction << "," << P_matrix_mean << ","
                           << P_fiber_mean << "," << P_shear_mean << "," << tracking_error_rms << ","
                           << tracking_error_max << ","
                           << tracking_error_rms / swimmer_data.geometry.length << ","
                           << tracking_error_max / swimmer_data.geometry.length << "," << body_force_x << ","
                           << body_force_y << "," << body_torque << "," << body_power << ","
                           << tail_root_lateral << "," << tail_tip_lateral << "," << tail_lateral_over_L << ","
                           << tail_pitch << "," << target_tail_root_lateral << ","
                           << target_tail_tip_lateral << "," << target_tail_pitch << ","
                           << actual_curvature_mid << "," << actual_curvature_tail << ","
                           << target_curvature_mid << "," << target_curvature_tail << "\n";
        diagnostics_stream.flush();
    }
}

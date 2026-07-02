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
enum class PrescribedMotionMode
{
    TETHERED,
    FREE_SWIMMING
};

PrescribedMotionMode
parse_prescribed_motion_mode(const std::string& value)
{
    if (value == "TETHERED") return PrescribedMotionMode::TETHERED;
    if (value == "FREE_SWIMMING") return PrescribedMotionMode::FREE_SWIMMING;
    TBOX_ERROR("PRESCRIBED_MOTION_MODE must be TETHERED or FREE_SWIMMING.\n");
    return PrescribedMotionMode::FREE_SWIMMING;
}

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

struct MaterialResponse
{
    TensorValue<double> total;
    TensorValue<double> matrix;
    TensorValue<double> fiber;
    TensorValue<double> shear;
    double J_material = 1.0;
};

struct RigidTargetFrame
{
    libMesh::Point origin;
    VectorValue<double> tangent;
    VectorValue<double> normal;
    VectorValue<double> origin_velocity;
    double angular_rate = 0.0;
    bool initialized = false;
};

struct SwimmerData
{
    PrescribedMotionMode prescribed_motion_mode;
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
    VectorValue<double> penalty_projection_translation;
    double penalty_projection_rotation = 0.0;
    libMesh::Point penalty_projection_center;
    RigidTargetFrame free_swimming_target_frame;
    RigidTargetFrame free_swimming_shape_frame;
    double free_swimming_shape_frame_time = std::numeric_limits<double>::quiet_NaN();

    double passive_mu;
    double passive_lambda;
    double passive_fiber_modulus;
    double passive_shear_modulus;
    double passive_material_scale;

    bool run_material_self_checks;
    double material_self_check_tolerance;
    double min_allowed_J;
    double max_allowed_J;
    bool abort_on_mesh_quality_limit;
    double min_allowed_target_J;
    double max_tracking_error_over_L;
    double max_tracking_velocity_error_over_Lf;
    bool write_midline_data;
    std::string midline_filename;
    std::string tracking_arclength_filename;
    std::string rigid_motion_filename;
    std::string penalty_projection_filename;
    int midline_num_stations;
    double midline_station_half_width;
    double midline_centerline_half_thickness;
    bool midline_use_body_frame;
    double midline_body_frame_head_xi;
    double midline_body_frame_tangent_xi;
    double body_frame_fit_xi_min;
    double body_frame_fit_xi_max;

    SwimmerData(Pointer<Database> input_db, const ReferenceGeometry& reference_geometry)
        : prescribed_motion_mode(parse_prescribed_motion_mode(
              input_db->getStringWithDefault("PRESCRIBED_MOTION_MODE", "FREE_SWIMMING"))),
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
          passive_material_scale(input_db->getDouble("PASSIVE_MATERIAL_SCALE")),
          run_material_self_checks(input_db->getBool("RUN_MATERIAL_SELF_CHECKS")),
          material_self_check_tolerance(input_db->getDouble("MATERIAL_SELF_CHECK_TOLERANCE")),
          min_allowed_J(input_db->getDouble("MIN_ALLOWED_J")),
          max_allowed_J(input_db->getDouble("MAX_ALLOWED_J")),
          abort_on_mesh_quality_limit(input_db->getBool("ABORT_ON_MESH_QUALITY_LIMIT")),
          min_allowed_target_J(input_db->getDoubleWithDefault("MIN_ALLOWED_TARGET_J", 0.50)),
          max_tracking_error_over_L(input_db->getDouble("MAX_TRACKING_ERROR_OVER_L")),
          max_tracking_velocity_error_over_Lf(
              input_db->getDouble("MAX_TRACKING_VELOCITY_ERROR_OVER_LF")),
          write_midline_data(input_db->getBoolWithDefault("WRITE_MIDLINE_DATA", true)),
          midline_filename(input_db->getStringWithDefault("MIDLINE_FILENAME",
                                                          "midline_target_actual.csv")),
          tracking_arclength_filename(input_db->getStringWithDefault(
              "TRACKING_ARCLENGTH_FILENAME", "tracking_arclength_diag.csv")),
          rigid_motion_filename(input_db->getStringWithDefault("RIGID_MOTION_FILENAME",
                                                               "rigid_motion_diag.csv")),
          penalty_projection_filename(input_db->getStringWithDefault(
              "PENALTY_PROJECTION_FILENAME", "penalty_projection_diag.csv")),
          midline_num_stations(input_db->getIntegerWithDefault("MIDLINE_NUM_STATIONS", 151)),
          midline_station_half_width(
              input_db->getDoubleWithDefault("MIDLINE_STATION_HALF_WIDTH", 0.008)),
          midline_centerline_half_thickness(
              input_db->getDoubleWithDefault("MIDLINE_CENTERLINE_HALF_THICKNESS", 0.003)),
          midline_use_body_frame(input_db->getBoolWithDefault("MIDLINE_USE_BODY_FRAME", true)),
          midline_body_frame_head_xi(
              input_db->getDoubleWithDefault("MIDLINE_BODY_FRAME_HEAD_XI", 0.02)),
          midline_body_frame_tangent_xi(
              input_db->getDoubleWithDefault("MIDLINE_BODY_FRAME_TANGENT_XI", 0.10)),
          body_frame_fit_xi_min(input_db->getDoubleWithDefault("BODY_FRAME_FIT_XI_MIN", 0.02)),
          body_frame_fit_xi_max(input_db->getDoubleWithDefault("BODY_FRAME_FIT_XI_MAX", 0.20))
    {
        if (geometry.length <= 0.0) TBOX_ERROR("The reference mesh has nonpositive body length.\n");
        if (frequency <= 0.0) TBOX_ERROR("WAVE_FREQUENCY must be positive.\n");
        if (wave_count <= 0.0) TBOX_ERROR("WAVE_COUNT must be positive.\n");
        if (ramp_time < 0.0) TBOX_ERROR("WAVE_RAMP_TIME must be nonnegative.\n");
        if (prescribed_head_amplitude < 0.0 || prescribed_tail_amplitude < 0.0)
        {
            TBOX_ERROR("Prescribed amplitudes must be nonnegative.\n");
        }
        {
            const bool p_is_one = std::abs(prescribed_envelope_power - 1.0) < 1.0e-12;
            if (!(p_is_one || prescribed_envelope_power >= 2.0))
            {
                TBOX_ERROR("PRESCRIBED_ENVELOPE_POWER must be 1 or >= 2 "
                           "because target curvature requires a finite envelope second derivative at s=0.\n");
            }
        }
        if (penalty_stiffness < 0.0 || penalty_damping < 0.0)
        {
            TBOX_ERROR("Penalty coefficients must be nonnegative.\n");
        }
        if (penalty_stiffness <= 0.0)
        {
            TBOX_ERROR("Prescribed penalty kinematics requires PENALTY_STIFFNESS > 0.\n");
        }
        if (passive_material_scale < 0.0)
        {
            TBOX_ERROR("PASSIVE_MATERIAL_SCALE must be nonnegative.\n");
        }
        if (passive_material_scale > 0.0 &&
            (passive_mu <= 0.0 || passive_lambda < 0.0 || passive_fiber_modulus < 0.0 ||
             passive_shear_modulus < 0.0))
        {
            TBOX_ERROR("Passive material coefficients are invalid.\n");
        }
        if (material_self_check_tolerance <= 0.0)
        {
            TBOX_ERROR("MATERIAL_SELF_CHECK_TOLERANCE must be positive.\n");
        }
        if (!(0.0 < min_allowed_J && min_allowed_J < 1.0 && 1.0 < max_allowed_J))
        {
            TBOX_ERROR("Require 0 < MIN_ALLOWED_J < 1 < MAX_ALLOWED_J.\n");
        }
        if (min_allowed_target_J <= 0.0)
        {
            TBOX_ERROR("MIN_ALLOWED_TARGET_J must be positive.\n");
        }
        if (max_tracking_error_over_L <= 0.0)
        {
            TBOX_ERROR("MAX_TRACKING_ERROR_OVER_L must be positive.\n");
        }
        if (max_tracking_velocity_error_over_Lf <= 0.0)
        {
            TBOX_ERROR("MAX_TRACKING_VELOCITY_ERROR_OVER_LF must be positive.\n");
        }
        if (midline_num_stations < 3)
        {
            TBOX_ERROR("MIDLINE_NUM_STATIONS must be at least 3.\n");
        }
        if (!(0.0 < midline_station_half_width && midline_station_half_width < 0.1))
        {
            TBOX_ERROR("MIDLINE_STATION_HALF_WIDTH must be in (0,0.1).\n");
        }
        if (midline_centerline_half_thickness <= 0.0)
        {
            TBOX_ERROR("MIDLINE_CENTERLINE_HALF_THICKNESS must be positive.\n");
        }
        if (!(0.0 <= midline_body_frame_head_xi && midline_body_frame_head_xi < 1.0 &&
              midline_body_frame_head_xi < midline_body_frame_tangent_xi &&
              midline_body_frame_tangent_xi <= 1.0))
        {
            TBOX_ERROR("Require 0 <= MIDLINE_BODY_FRAME_HEAD_XI < "
                       "MIDLINE_BODY_FRAME_TANGENT_XI <= 1.\n");
        }
        if (!(0.0 <= body_frame_fit_xi_min && body_frame_fit_xi_min < body_frame_fit_xi_max &&
              body_frame_fit_xi_max <= 1.0))
        {
            TBOX_ERROR("Require 0 <= BODY_FRAME_FIT_XI_MIN < BODY_FRAME_FIT_XI_MAX <= 1.\n");
        }
        penalty_projection_translation.zero();
        penalty_projection_center = geometry.head + 0.5 * geometry.length * geometry.tangent;
        free_swimming_target_frame.origin = geometry.head;
        const double frame_fit_center_xi = 0.5 * (body_frame_fit_xi_min + body_frame_fit_xi_max);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            free_swimming_target_frame.origin(d) +=
                frame_fit_center_xi * geometry.length * geometry.tangent(d);
        }
        free_swimming_target_frame.tangent = geometry.tangent;
        free_swimming_target_frame.normal = geometry.normal;
        free_swimming_target_frame.origin_velocity.zero();
        free_swimming_target_frame.angular_rate = 0.0;
        free_swimming_target_frame.initialized = true;
        free_swimming_shape_frame.initialized = false;
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

struct BodyFrameFitResult
{
    RigidTargetFrame actual_frame;
    RigidTargetFrame target_shape_frame;
};

struct MidlineSample
{
    double xi = 0.0;
    libMesh::Point position;
    VectorValue<double> velocity;
    std::string source;
};

struct MidlineFrame
{
    libMesh::Point origin;
    VectorValue<double> tangent;
    VectorValue<double> normal;
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

double
prescribed_amplitude_second_derivative(const double s, const SwimmerData& data)
{
    if (data.prescribed_envelope_power <= 1.0) return 0.0;
    const double xi = unit_coordinate(s, data);
    const double delta = data.prescribed_tail_amplitude - data.prescribed_head_amplitude;
    return delta * data.prescribed_envelope_power * (data.prescribed_envelope_power - 1.0) *
           std::pow(xi, data.prescribed_envelope_power - 2.0) /
           (data.geometry.length * data.geometry.length);
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
target_shape_state(const double s, const double eta, const double time, const SwimmerData& data)
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
    target.position(0) = local_x;
    target.position(1) = local_y;
    target.velocity(0) = local_velocity_x;
    target.velocity(1) = local_velocity_y;
    return target;
}

TargetState
target_shape_state(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    return target_shape_state(reference_s(X, data), reference_eta(X, data), time, data);
}

RigidTargetFrame
make_shape_frame(const double time, const SwimmerData& data)
{
    constexpr unsigned int num_fit_points = 33;
    double weight_sum = 0.0;
    double s_sum = 0.0;
    double x_sum = 0.0;
    double y_sum = 0.0;
    double u_sum = 0.0;
    double v_sum = 0.0;
    std::array<TargetState, num_fit_points> states;
    std::array<double, num_fit_points> s_values;
    for (unsigned int i = 0; i < num_fit_points; ++i)
    {
        const double alpha = static_cast<double>(i) / static_cast<double>(num_fit_points - 1);
        const double xi =
            data.body_frame_fit_xi_min +
            alpha * (data.body_frame_fit_xi_max - data.body_frame_fit_xi_min);
        const double s = xi * data.geometry.length;
        const double weight = (i == 0 || i + 1 == num_fit_points) ? 0.5 : 1.0;
        states[i] = target_shape_state(s, 0.0, time, data);
        s_values[i] = s;
        weight_sum += weight;
        s_sum += weight * s;
        x_sum += weight * states[i].position(0);
        y_sum += weight * states[i].position(1);
        u_sum += weight * states[i].velocity(0);
        v_sum += weight * states[i].velocity(1);
    }
    if (!(weight_sum > 0.0))
    {
        TBOX_ERROR("Cannot define prescribed target shape frame: no fit points were sampled.\n");
    }
    const double s_centroid = s_sum / weight_sum;
    RigidTargetFrame frame;
    frame.origin(0) = x_sum / weight_sum;
    frame.origin(1) = y_sum / weight_sum;
    frame.origin_velocity(0) = u_sum / weight_sum;
    frame.origin_velocity(1) = v_sum / weight_sum;

    double cos_sum = 0.0;
    double sin_sum = 0.0;
    double angular_numerator = 0.0;
    double angular_denominator = 0.0;
    for (unsigned int i = 0; i < num_fit_points; ++i)
    {
        const double weight = (i == 0 || i + 1 == num_fit_points) ? 0.5 : 1.0;
        const double a = s_values[i] - s_centroid;
        const VectorValue<double> radius = states[i].position - frame.origin;
        const VectorValue<double> relative_velocity = states[i].velocity - frame.origin_velocity;
        cos_sum += weight * a * radius(0);
        sin_sum += weight * a * radius(1);
        angular_numerator +=
            weight * (radius(0) * relative_velocity(1) - radius(1) * relative_velocity(0));
        angular_denominator += weight * dot2(radius, radius);
    }
    const double tangent_norm = std::sqrt(cos_sum * cos_sum + sin_sum * sin_sum);
    if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm))
    {
        TBOX_ERROR("Cannot define prescribed target shape frame: fitted tangent is invalid.\n");
    }
    frame.tangent(0) = cos_sum / tangent_norm;
    frame.tangent(1) = sin_sum / tangent_norm;
    frame.normal(0) = -frame.tangent(1);
    frame.normal(1) = frame.tangent(0);
    frame.angular_rate =
        angular_denominator > 0.0 ? angular_numerator / angular_denominator : 0.0;
    frame.initialized = true;
    return frame;
}

RigidTargetFrame
make_reference_target_frame(const SwimmerData& data)
{
    RigidTargetFrame frame;
    frame.origin = data.geometry.head;
    frame.tangent = data.geometry.tangent;
    frame.normal = data.geometry.normal;
    frame.origin_velocity.zero();
    frame.angular_rate = 0.0;
    frame.initialized = true;
    return frame;
}

TargetState
place_target_shape_state(const TargetState& shape_state,
                         const RigidTargetFrame& shape_frame,
                         const RigidTargetFrame& lab_frame)
{
    if (!shape_frame.initialized || !lab_frame.initialized)
    {
        TBOX_ERROR("Cannot place prescribed target shape with an uninitialized frame.\n");
    }

    const VectorValue<double> shape_radius = shape_state.position - shape_frame.origin;
    const double body_x = dot2(shape_radius, shape_frame.tangent);
    const double body_y = dot2(shape_radius, shape_frame.normal);

    VectorValue<double> lab_radius;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        lab_radius(d) = body_x * lab_frame.tangent(d) + body_y * lab_frame.normal(d);
    }

    VectorValue<double> shape_rigid_velocity;
    shape_rigid_velocity(0) =
        shape_frame.origin_velocity(0) - shape_frame.angular_rate * shape_radius(1);
    shape_rigid_velocity(1) =
        shape_frame.origin_velocity(1) + shape_frame.angular_rate * shape_radius(0);
    const VectorValue<double> shape_deformation_velocity =
        shape_state.velocity - shape_rigid_velocity;
    const double deformation_u = dot2(shape_deformation_velocity, shape_frame.tangent);
    const double deformation_v = dot2(shape_deformation_velocity, shape_frame.normal);

    VectorValue<double> lab_rigid_velocity;
    lab_rigid_velocity(0) =
        lab_frame.origin_velocity(0) - lab_frame.angular_rate * lab_radius(1);
    lab_rigid_velocity(1) =
        lab_frame.origin_velocity(1) + lab_frame.angular_rate * lab_radius(0);

    TargetState target;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        target.position(d) = lab_frame.origin(d) + lab_radius(d);
        target.velocity(d) = lab_rigid_velocity(d) +
                             deformation_u * lab_frame.tangent(d) +
                             deformation_v * lab_frame.normal(d);
    }
    return target;
}

TargetState
target_state(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    const TargetState shape_state = target_shape_state(X, time, data);
    if (data.prescribed_motion_mode == PrescribedMotionMode::FREE_SWIMMING)
    {
        const bool have_cached_shape_frame =
            data.free_swimming_shape_frame.initialized &&
            std::abs(data.free_swimming_shape_frame_time - time) <=
                10.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(time));
        return place_target_shape_state(shape_state,
                                        have_cached_shape_frame ?
                                            data.free_swimming_shape_frame :
                                            make_shape_frame(time, data),
                                        data.free_swimming_target_frame);
    }

    TargetState target;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        target.position(d) = data.geometry.head(d) +
                             shape_state.position(0) * data.geometry.tangent(d) +
                             shape_state.position(1) * data.geometry.normal(d);
        target.velocity(d) = shape_state.velocity(0) * data.geometry.tangent(d) +
                             shape_state.velocity(1) * data.geometry.normal(d);
    }
    return target;
}

double
target_curvature_at_s(const double s, const double time, const SwimmerData& data)
{
    double ramp_value = 0.0;
    double ramp_derivative = 0.0;
    ramp(time, data, ramp_value, ramp_derivative);

    const double k = wave_number(data);
    const double phase = k * s - angular_frequency(data) * time + data.phase0;
    double amplitude = 0.0;
    double amplitude_s = 0.0;
    prescribed_amplitude(s, data, amplitude, amplitude_s);
    const double amplitude_ss = prescribed_amplitude_second_derivative(s, data);
    const double h_s = amplitude_s * std::sin(phase) + amplitude * k * std::cos(phase);
    const double h_ss = amplitude_ss * std::sin(phase) +
                        2.0 * amplitude_s * k * std::cos(phase) -
                        amplitude * k * k * std::sin(phase);
    const double slope = ramp_value * h_s;
    return ramp_value * h_ss / (1.0 + slope * slope);
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

MaterialResponse
evaluate_material(const TensorValue<double>& FF,
                  const SwimmerData& data,
                  const libMesh::Point* X,
                  const double time)
{
    MaterialResponse response;
    response.J_material = FF(0, 0) * FF(1, 1) - FF(0, 1) * FF(1, 0);
    if (!(response.J_material > 0.0) || !std::isfinite(response.J_material))
    {
        if (X)
        {
            TBOX_ERROR("Nonpositive material Jacobian at time "
                       << time << ", X=(" << (*X)(0) << "," << (*X)(1)
                       << "), J=" << response.J_material << "\n");
        }
        TBOX_ERROR("Nonpositive material Jacobian in material self-check: Je="
                   << response.J_material << "\n");
    }

    const double material_scale = data.passive_material_scale;
    const TensorValue<double> FF_inverse_transpose = tensor_inverse_transpose(FF, NDIM);
    TensorValue<double> Pe_matrix;
    Pe_matrix.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            Pe_matrix(i, j) =
                material_scale *
                (data.passive_mu * (FF(i, j) - FF_inverse_transpose(i, j)) +
                 data.passive_lambda * std::log(response.J_material) * FF_inverse_transpose(i, j));
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
            fiber(i) += FF(i, j) * data.geometry.tangent(j);
            transverse(i) += FF(i, j) * data.geometry.normal(j);
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

    response.matrix = Pe_matrix;
    response.fiber = Pe_fiber;
    response.shear = Pe_shear;
    response.total = response.matrix + response.fiber + response.shear;
    return response;
}

void
PK1_material_function(TensorValue<double>& PP,
                      const TensorValue<double>& FF,
                      const libMesh::Point&,
                      const libMesh::Point& X,
                      Elem* const,
                      const vector<const vector<double>*>&,
                      const vector<const vector<VectorValue<double> >*>&,
                      const double time,
                      void* ctx)
{
    const auto& data = *static_cast<SwimmerData*>(ctx);
    PP = evaluate_material(FF, data, &X, time).total;
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
    VectorValue<double> velocity_error = target.velocity - velocity;
    if (data.prescribed_motion_mode == PrescribedMotionMode::TETHERED)
    {
        const double characteristic_velocity = data.geometry.length * data.frequency;
        const double velocity_error_over_Lf =
            std::sqrt(dot2(velocity_error, velocity_error)) / characteristic_velocity;
        if (velocity_error_over_Lf > data.max_tracking_velocity_error_over_Lf)
        {
            TBOX_ERROR("Penalty velocity tracking limit exceeded at time = "
                       << time << ": |U_target-U|/(L*f) = " << velocity_error_over_Lf
                       << ", allowed maximum = " << data.max_tracking_velocity_error_over_Lf << ".\n");
        }
    }
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        F(d) = data.penalty_stiffness * (target.position(d) - x(d)) +
               data.penalty_damping * velocity_error(d);
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
    raw_penalty_force(F, x, X, velocity, time, data);
    if (data.prescribed_motion_mode == PrescribedMotionMode::FREE_SWIMMING)
    {
        F -= data.penalty_projection_translation;
        const VectorValue<double> radius = x - data.penalty_projection_center;
        F(0) += data.penalty_projection_rotation * radius(1);
        F(1) -= data.penalty_projection_rotation * radius(0);
    }
}

void
lag_body_force_function(VectorValue<double>& F,
                        const TensorValue<double>&,
                        const libMesh::Point& x,
                        const libMesh::Point& X,
                        Elem* const,
                        const vector<const vector<double>*>& var_data,
                        const vector<const vector<VectorValue<double> >*>&,
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

    const double eta_center = 0.5 * (eta_min + eta_max);
    ReferenceGeometry geometry;
    geometry.tangent = tangent;
    geometry.normal = normal;
    geometry.length = projection_max - projection_min;
    geometry.eta_min = eta_min - eta_center;
    geometry.eta_max = eta_max - eta_center;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        geometry.head(d) = head_projection * axis(d) + eta_center * normal(d);
    }
    return geometry;
}

RigidTargetFrame
make_fitted_frame(const libMesh::Point& origin,
                  const VectorValue<double>& origin_velocity,
                  const double cos_sum,
                  const double sin_sum,
                  const double angular_numerator,
                  const double angular_denominator,
                  const std::string& frame_name,
                  const double frame_time)
{
    RigidTargetFrame frame;
    frame.origin = origin;
    frame.origin_velocity = origin_velocity;
    const double tangent_norm = std::sqrt(cos_sum * cos_sum + sin_sum * sin_sum);
    if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm))
    {
        TBOX_ERROR("Cannot fit " << frame_name << " body frame at time = " << frame_time
                                 << ": fitted tangent is invalid.\n");
    }
    frame.tangent(0) = cos_sum / tangent_norm;
    frame.tangent(1) = sin_sum / tangent_norm;
    frame.normal(0) = -frame.tangent(1);
    frame.normal(1) = frame.tangent(0);
    frame.angular_rate =
        angular_denominator > 0.0 ? angular_numerator / angular_denominator : 0.0;
    frame.initialized = true;
    return frame;
}

BodyFrameFitResult
fit_body_frames_from_quadrature(Mesh& mesh,
                                EquationSystems& equation_systems,
                                const std::string& coords_system_name,
                                const std::string& velocity_system_name,
                                const SwimmerData& data,
                                const double frame_time)
{
    System& X_system = equation_systems.get_system(coords_system_name);
    System& U_system = equation_systems.get_system(velocity_system_name);
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

    std::array<double, 11> centroid_sums = { 0.0 };
    for (auto elem_it = mesh.active_local_elements_begin(); elem_it != mesh.active_local_elements_end(); ++elem_it)
    {
        Elem* const elem = *elem_it;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d) dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);
        get_values_for_interpolation(U_node, *U_ghost_vec, dof_indices);
        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            const libMesh::Point& X = reference_points[qp];
            const double s = reference_s(X, data);
            const double xi = unit_coordinate(s, data);
            if (xi < data.body_frame_fit_xi_min || xi > data.body_frame_fit_xi_max) continue;
            interpolate(x, qp, X_node, phi);
            interpolate(velocity, qp, U_node, phi);
            const double eta = reference_eta(X, data);
            const TargetState shape = target_shape_state(s, eta, frame_time, data);
            const double weight = JxW[qp];
            centroid_sums[0] += weight;
            centroid_sums[1] += weight * s;
            centroid_sums[2] += weight * eta;
            centroid_sums[3] += weight * x(0);
            centroid_sums[4] += weight * x(1);
            centroid_sums[5] += weight * velocity(0);
            centroid_sums[6] += weight * velocity(1);
            centroid_sums[7] += weight * shape.position(0);
            centroid_sums[8] += weight * shape.position(1);
            centroid_sums[9] += weight * shape.velocity(0);
            centroid_sums[10] += weight * shape.velocity(1);
        }
    }
    IBTK_MPI::sumReduction(centroid_sums.data(), centroid_sums.size());

    const double weight_sum = centroid_sums[0];
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum))
    {
        TBOX_ERROR("Cannot fit body frame at time = " << frame_time
                                                      << ": no quadrature points in BODY_FRAME_FIT_XI range ["
                                                      << data.body_frame_fit_xi_min << ","
                                                      << data.body_frame_fit_xi_max << "].\n");
    }

    const double ref_s_centroid = centroid_sums[1] / weight_sum;
    const double ref_eta_centroid = centroid_sums[2] / weight_sum;
    libMesh::Point actual_origin;
    actual_origin(0) = centroid_sums[3] / weight_sum;
    actual_origin(1) = centroid_sums[4] / weight_sum;
    VectorValue<double> actual_origin_velocity;
    actual_origin_velocity(0) = centroid_sums[5] / weight_sum;
    actual_origin_velocity(1) = centroid_sums[6] / weight_sum;
    libMesh::Point shape_origin;
    shape_origin(0) = centroid_sums[7] / weight_sum;
    shape_origin(1) = centroid_sums[8] / weight_sum;
    VectorValue<double> shape_origin_velocity;
    shape_origin_velocity(0) = centroid_sums[9] / weight_sum;
    shape_origin_velocity(1) = centroid_sums[10] / weight_sum;

    std::array<double, 8> fit_sums = { 0.0 };
    for (auto elem_it = mesh.active_local_elements_begin(); elem_it != mesh.active_local_elements_end(); ++elem_it)
    {
        Elem* const elem = *elem_it;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d) dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);
        get_values_for_interpolation(U_node, *U_ghost_vec, dof_indices);
        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            const libMesh::Point& X = reference_points[qp];
            const double s = reference_s(X, data);
            const double xi = unit_coordinate(s, data);
            if (xi < data.body_frame_fit_xi_min || xi > data.body_frame_fit_xi_max) continue;
            interpolate(x, qp, X_node, phi);
            interpolate(velocity, qp, U_node, phi);
            const double eta = reference_eta(X, data);
            const TargetState shape = target_shape_state(s, eta, frame_time, data);
            const double a_s = s - ref_s_centroid;
            const double a_eta = eta - ref_eta_centroid;
            const double weight = JxW[qp];

            const VectorValue<double> actual_radius = x - actual_origin;
            const VectorValue<double> actual_relative_velocity =
                velocity - actual_origin_velocity;
            fit_sums[0] += weight * (a_s * actual_radius(0) + a_eta * actual_radius(1));
            fit_sums[1] += weight * (a_s * actual_radius(1) - a_eta * actual_radius(0));
            fit_sums[2] += weight * (actual_radius(0) * actual_relative_velocity(1) -
                                     actual_radius(1) * actual_relative_velocity(0));
            fit_sums[3] += weight * dot2(actual_radius, actual_radius);

            const VectorValue<double> shape_radius = shape.position - shape_origin;
            const VectorValue<double> shape_relative_velocity =
                shape.velocity - shape_origin_velocity;
            fit_sums[4] += weight * (a_s * shape_radius(0) + a_eta * shape_radius(1));
            fit_sums[5] += weight * (a_s * shape_radius(1) - a_eta * shape_radius(0));
            fit_sums[6] += weight * (shape_radius(0) * shape_relative_velocity(1) -
                                     shape_radius(1) * shape_relative_velocity(0));
            fit_sums[7] += weight * dot2(shape_radius, shape_radius);
        }
    }
    IBTK_MPI::sumReduction(fit_sums.data(), fit_sums.size());

    BodyFrameFitResult result;
    result.actual_frame = make_fitted_frame(actual_origin,
                                            actual_origin_velocity,
                                            fit_sums[0],
                                            fit_sums[1],
                                            fit_sums[2],
                                            fit_sums[3],
                                            "actual",
                                            frame_time);
    result.target_shape_frame = make_fitted_frame(shape_origin,
                                                  shape_origin_velocity,
                                                  fit_sums[4],
                                                  fit_sums[5],
                                                  fit_sums[6],
                                                  fit_sums[7],
                                                  "target-shape",
                                                  frame_time);
    return result;
}

void
update_free_swimming_target_frame(Mesh& mesh,
                                  EquationSystems& equation_systems,
                                  const std::string& coords_system_name,
                                  const std::string& velocity_system_name,
                                  SwimmerData& data,
                                  const double frame_time)
{
    const BodyFrameFitResult fitted_frames = fit_body_frames_from_quadrature(
        mesh, equation_systems, coords_system_name, velocity_system_name, data, frame_time);
    data.free_swimming_shape_frame = fitted_frames.target_shape_frame;
    data.free_swimming_shape_frame_time = frame_time;
    data.free_swimming_target_frame = fitted_frames.actual_frame;
}

void
update_penalty_projection(const double current_time,
                          const double new_time,
                          const int,
                          void* ctx)
{
    auto& projection = *static_cast<PenaltyProjectionContext*>(ctx);
    SwimmerData& data = *projection.swimmer_data;
    data.penalty_projection_translation.zero();
    data.penalty_projection_rotation = 0.0;

    Mesh& mesh = *projection.mesh;
    EquationSystems& equation_systems = *projection.equation_systems;
    const double projection_time = 0.5 * (current_time + new_time);
    update_free_swimming_target_frame(mesh,
                                      equation_systems,
                                      projection.coords_system_name,
                                      projection.velocity_system_name,
                                      data,
                                      projection_time);

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
        data.penalty_projection_translation(d) = force_integral[d] / volume;
    }

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
            raw_force -= data.penalty_projection_translation;
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
postprocess_penalty_projection(const double,
                               const double new_time,
                               const bool,
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
    const MaterialResponse passive = evaluate_material(identity, data, nullptr, 0.0);
    const double passive_residual = frobenius_norm(passive.total);

    pout << "Material self-check: ||P(F=I)|| = " << passive_residual << "\n";
    if (!std::isfinite(passive_residual) ||
        passive_residual > data.material_self_check_tolerance)
    {
        TBOX_ERROR("Passive material self-check failed. Tolerance = "
                   << data.material_self_check_tolerance << "\n");
    }
}

void
check_target_mapping_quality(const SwimmerData& data)
{
    constexpr unsigned int num_space_samples = 512;
    constexpr unsigned int num_time_samples = 512;
    const double end_time = data.ramp_time + 1.0 / data.frequency;
    double J_min = std::numeric_limits<double>::max();
    double kappa_abs_max = 0.0;
    for (unsigned int ti = 0; ti <= num_time_samples; ++ti)
    {
        const double time = end_time * static_cast<double>(ti) / num_time_samples;
        for (unsigned int si = 0; si <= num_space_samples; ++si)
        {
            const double s = data.geometry.length * static_cast<double>(si) / num_space_samples;
            const double kappa = target_curvature_at_s(s, time, data);
            kappa_abs_max = std::max(kappa_abs_max, std::abs(kappa));
            for (const double eta : { data.geometry.eta_min, data.geometry.eta_max })
            {
                J_min = std::min(J_min, 1.0 - eta * kappa);
            }
        }
    }
    pout << "Target mapping scan: min(1-eta*kappa_target) = " << J_min
         << ", max |kappa_target|*L = " << kappa_abs_max * data.geometry.length << "\n";
    if (!(J_min > data.min_allowed_target_J) || !std::isfinite(J_min))
    {
        TBOX_ERROR("Target mapping quality limit failed: min(1-eta*kappa_target) = "
                   << J_min << ", required > " << data.min_allowed_target_J << ".\n");
    }
}
} // namespace ModelData
using namespace ModelData;

static std::ofstream midline_stream;
static std::ofstream tracking_arclength_stream;
static std::ofstream rigid_motion_stream;
static std::ofstream penalty_projection_stream;

std::string
join_output_path(const std::string& directory, const std::string& filename)
{
    if (filename.empty() || filename[0] == '/' || directory.empty()) return filename;
    if (directory.back() == '/') return directory + filename;
    return directory + "/" + filename;
}

libMesh::Point
reference_centerline_point(const double xi, const SwimmerData& swimmer_data)
{
    libMesh::Point point = swimmer_data.geometry.head;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        point(d) += xi * swimmer_data.geometry.length * swimmer_data.geometry.tangent(d);
    }
    return point;
}

MidlineFrame
make_midline_frame(const libMesh::Point& origin,
                   const libMesh::Point& tangent_point)
{
    MidlineFrame frame;
    frame.origin = origin;
    frame.tangent = tangent_point - origin;
    const double tangent_norm = std::sqrt(dot2(frame.tangent, frame.tangent));
    if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm))
    {
        TBOX_ERROR("Cannot define midline body frame: the head and tangent stations are "
                   "coincident or invalid.\n");
    }
    frame.tangent /= tangent_norm;
    frame.normal(0) = -frame.tangent(1);
    frame.normal(1) = frame.tangent(0);
    return frame;
}

MidlineFrame
make_reference_midline_frame(const SwimmerData& swimmer_data)
{
    MidlineFrame frame;
    frame.origin = swimmer_data.geometry.head;
    frame.tangent = swimmer_data.geometry.tangent;
    frame.normal = swimmer_data.geometry.normal;
    return frame;
}

MidlineFrame
make_midline_frame_from_rigid_frame(const RigidTargetFrame& rigid_frame)
{
    MidlineFrame frame;
    frame.origin = rigid_frame.origin;
    frame.tangent = rigid_frame.tangent;
    frame.normal = rigid_frame.normal;
    return frame;
}

std::array<double, 2>
body_frame_coordinates(const libMesh::Point& point, const MidlineFrame& frame)
{
    const VectorValue<double> displacement = point - frame.origin;
    return { dot2(displacement, frame.tangent), dot2(displacement, frame.normal) };
}

std::array<double, 2>
deformation_velocity_components(const libMesh::Point& point,
                                const VectorValue<double>& velocity,
                                const RigidTargetFrame& frame)
{
    const VectorValue<double> radius = point - frame.origin;
    VectorValue<double> rigid_velocity;
    rigid_velocity(0) = frame.origin_velocity(0) - frame.angular_rate * radius(1);
    rigid_velocity(1) = frame.origin_velocity(1) + frame.angular_rate * radius(0);
    const VectorValue<double> deformation_velocity = velocity - rigid_velocity;
    return { dot2(deformation_velocity, frame.tangent),
             dot2(deformation_velocity, frame.normal) };
}

double
point_curvature(const libMesh::Point& a, const libMesh::Point& b, const libMesh::Point& c)
{
    const double ab = (b - a).norm();
    const double bc = (c - b).norm();
    const double ac = (c - a).norm();
    const double denominator = ab * bc * ac;
    if (denominator <= std::numeric_limits<double>::epsilon()) return 0.0;
    const double cross = (b(0) - a(0)) * (c(1) - a(1)) -
                         (b(1) - a(1)) * (c(0) - a(0));
    return 2.0 * cross / denominator;
}

double
midline_tangent_angle(const std::vector<libMesh::Point>& points,
                      const std::size_t index,
                      const MidlineFrame& frame)
{
    const std::size_t lo = index == 0 ? index : index - 1;
    const std::size_t hi = index + 1 < points.size() ? index + 1 : index;
    const VectorValue<double> direction = points[hi] - points[lo];
    return std::atan2(dot2(direction, frame.normal), dot2(direction, frame.tangent));
}

double
midline_curvature(const std::vector<libMesh::Point>& points, const std::size_t index)
{
    if (points.size() < 3) return 0.0;
    const std::size_t i0 = index == 0 ? 0 : index - 1;
    const std::size_t i1 = index;
    const std::size_t i2 = index + 1 < points.size() ? index + 1 : points.size() - 1;
    if (i0 == i1 || i1 == i2) return 0.0;
    return point_curvature(points[i0], points[i1], points[i2]);
}

int
assemble_midline_samples(const std::vector<double>& section_weight,
                         const std::vector<double>& section_x,
                         const std::vector<double>& section_y,
                         const std::vector<double>& section_u,
                         const std::vector<double>& section_v,
                         const std::vector<double>& centerline_weight,
                         const std::vector<double>& centerline_x,
                         const std::vector<double>& centerline_y,
                         const std::vector<double>& centerline_u,
                         const std::vector<double>& centerline_v,
                         const std::vector<double>& station_xi,
                         const int iteration_num,
                         const double loop_time,
                         const SwimmerData& swimmer_data,
                         std::vector<MidlineSample>& samples)
{
    const std::size_t num_stations = station_xi.size();
    samples.assign(num_stations, MidlineSample());
    int fallback_count = 0;
    for (std::size_t station = 0; station < num_stations; ++station)
    {
        samples[station].xi = station_xi[station];
        if (centerline_weight[station] > 0.0)
        {
            samples[station].position(0) = centerline_x[station] / centerline_weight[station];
            samples[station].position(1) = centerline_y[station] / centerline_weight[station];
            samples[station].velocity(0) = centerline_u[station] / centerline_weight[station];
            samples[station].velocity(1) = centerline_v[station] / centerline_weight[station];
            samples[station].source = "ETA0";
        }
        else if (section_weight[station] > 0.0)
        {
            samples[station].position(0) = section_x[station] / section_weight[station];
            samples[station].position(1) = section_y[station] / section_weight[station];
            samples[station].velocity(0) = section_u[station] / section_weight[station];
            samples[station].velocity(1) = section_v[station] / section_weight[station];
            samples[station].source = "SECTION_CENTROID_FALLBACK";
            ++fallback_count;
        }
        else
        {
            TBOX_ERROR("Unable to extract midline station at step "
                       << iteration_num << ", time = " << loop_time << ", station_index = "
                       << station << ", s/L = " << station_xi[station]
                       << ": no quadrature points in MIDLINE_STATION_HALF_WIDTH = "
                       << swimmer_data.midline_station_half_width
                       << ". Increase the station half-width or check the reference mesh.\n");
        }
    }
    return fallback_count;
}

void
write_midline_output(const std::vector<MidlineSample>& samples,
                     const std::vector<double>& station_xi,
                     const std::size_t frame_head_index,
                     const std::size_t frame_tangent_index,
                     const int num_output_stations,
                     const int fallback_count,
                     const SwimmerData& swimmer_data,
                     const int iteration_num,
                     const double loop_time)
{
    if (!swimmer_data.write_midline_data) return;

    MidlineFrame actual_frame = make_reference_midline_frame(swimmer_data);
    if (swimmer_data.midline_use_body_frame)
    {
        actual_frame = make_midline_frame(samples[frame_head_index].position,
                                          samples[frame_tangent_index].position);
    }

    std::vector<libMesh::Point> actual_points(num_output_stations);
    std::vector<libMesh::Point> target_points_all(station_xi.size());
    std::vector<libMesh::Point> target_points(num_output_stations);
    for (std::size_t station = 0; station < station_xi.size(); ++station)
    {
        const libMesh::Point X = reference_centerline_point(station_xi[station], swimmer_data);
        target_points_all[station] = target_state(X, loop_time, swimmer_data).position;
    }
    for (int station = 0; station < num_output_stations; ++station)
    {
        const std::size_t index = static_cast<std::size_t>(station);
        actual_points[index] = samples[index].position;
        target_points[index] = target_points_all[index];
    }
    MidlineFrame target_frame = make_reference_midline_frame(swimmer_data);
    if (swimmer_data.midline_use_body_frame)
    {
        target_frame = make_midline_frame(target_points_all[frame_head_index],
                                          target_points_all[frame_tangent_index]);
    }

    if (IBTK_MPI::getRank() != 0) return;

    if (fallback_count > 0)
    {
        const double fallback_fraction =
            static_cast<double>(fallback_count) / num_output_stations;
        pout << "Midline fallback at step " << iteration_num << ": " << fallback_count
             << "/" << num_output_stations << " stations ("
             << static_cast<int>(std::round(100.0 * fallback_fraction)) << "%) used section centroid.\n";
    }

    if (!midline_stream.is_open())
    {
        TBOX_ERROR("MIDLINE_FILENAME output stream is not open: "
                   << swimmer_data.midline_filename << "\n");
    }

    midline_stream.setf(std::ios::scientific);
    midline_stream.precision(12);
    for (int station = 0; station < num_output_stations; ++station)
    {
        const std::size_t index = static_cast<std::size_t>(station);
        const std::array<double, 2> actual_body =
            body_frame_coordinates(samples[index].position, actual_frame);
        const std::array<double, 2> target_body =
            body_frame_coordinates(target_points[index], target_frame);
        const double theta_actual = midline_tangent_angle(actual_points, index, actual_frame);
        const double theta_target = midline_tangent_angle(target_points, index, target_frame);
        const double kappa_actual = midline_curvature(actual_points, index);
        const double kappa_target = midline_curvature(target_points, index);
        midline_stream << loop_time << "," << samples[index].xi << ","
                       << target_body[0] << "," << target_body[1] << ","
                       << actual_body[0] << "," << actual_body[1] << ","
                       << samples[index].position(0) << "," << samples[index].position(1) << ","
                       << theta_target << "," << theta_actual << ","
                       << kappa_target << "," << kappa_actual << "\n";
    }
    midline_stream.flush();
}

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
        if (!postproc_data_dump_dirname.empty())
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
        pout << "=== Prescribed arclength-consistent kinematics via penalty body force ===\n";
        pout << "Passive material scale = " << swimmer_data.passive_material_scale << ".\n";
        if (swimmer_data.prescribed_motion_mode == PrescribedMotionMode::TETHERED)
        {
            pout << "Prescribed motion mode = TETHERED. The penalty target is fixed in the lab frame.\n";
        }
        else
        {
            pout << "Prescribed motion mode = FREE_SWIMMING. Net penalty force and torque are "
                    "projected out each step, and the target shape is placed in the "
                    "instantaneous body frame so the fish can recoil and swim.\n";
        }
        if (swimmer_data.passive_material_scale > 0.0)
        {
            run_material_self_checks(swimmer_data);
        }
        else
        {
            pout << "Passive material stress is disabled; only prescribed penalty actuation is active.\n";
        }
        check_target_mapping_quality(swimmer_data);

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

        if (swimmer_data.passive_material_scale > 0.0)
        {
            IBFEMethod::PK1StressFcnData PK1_stress_data(
                PK1_material_function, vector<SystemData>(), swimmer_data_ptr);
            PK1_stress_data.quad_order =
                Utility::string_to_enum<libMesh::Order>(input_db->getString("PK1_QUAD_ORDER"));
            ib_method_ops->registerPK1StressFunction(PK1_stress_data);
        }

        vector<int> velocity_variables(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d) velocity_variables[d] = d;
        vector<SystemData> system_data(1, SystemData(velocity_system_name, velocity_variables));
        IBFEMethod::LagBodyForceFcnData body_force_data(
            lag_body_force_function, system_data, swimmer_data_ptr);
        ib_method_ops->registerLagBodyForceFunction(body_force_data);
        if (swimmer_data.prescribed_motion_mode == PrescribedMotionMode::FREE_SWIMMING)
        {
            time_integrator->registerPreprocessIntegrateHierarchyCallback(
                update_penalty_projection, &penalty_projection_context);
            time_integrator->registerIntegrateHierarchyCallback(
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
        if (swimmer_data.prescribed_motion_mode == PrescribedMotionMode::FREE_SWIMMING)
        {
            const double initial_time = time_integrator->getIntegratorTime();
            update_penalty_projection(initial_time, initial_time, 1, &penalty_projection_context);
        }

        const bool from_restart = RestartManager::getManager()->isFromRestart();
        if (IBTK_MPI::getRank() == 0)
        {
            const auto output_path = [&](const std::string& filename) {
                return join_output_path(postproc_data_dump_dirname, filename);
            };
            if (swimmer_data.write_midline_data)
            {
                const std::string midline_path = output_path(swimmer_data.midline_filename);
                midline_stream.open(midline_path,
                                    from_restart ? ios_base::out | ios_base::app :
                                                   ios_base::out | ios_base::trunc);
                if (!midline_stream.is_open())
                {
                    TBOX_ERROR("Unable to open MIDLINE_FILENAME: " << midline_path << "\n");
                }
                midline_stream.precision(12);
                if (!from_restart)
                {
                    midline_stream
                        << "time,s_norm,x_target,y_target,x_body,y_body,x_lab,y_lab,"
                           "theta_target,theta_actual,kappa_target,kappa_actual\n";
                }
            }
            const std::string tracking_arclength_path =
                output_path(swimmer_data.tracking_arclength_filename);
            const std::string rigid_motion_path = output_path(swimmer_data.rigid_motion_filename);
            const std::string penalty_projection_path =
                output_path(swimmer_data.penalty_projection_filename);
            tracking_arclength_stream.open(
                tracking_arclength_path,
                from_restart ? ios_base::out | ios_base::app : ios_base::out | ios_base::trunc);
            rigid_motion_stream.open(
                rigid_motion_path,
                from_restart ? ios_base::out | ios_base::app : ios_base::out | ios_base::trunc);
            penalty_projection_stream.open(
                penalty_projection_path,
                from_restart ? ios_base::out | ios_base::app : ios_base::out | ios_base::trunc);
            if (!tracking_arclength_stream.is_open() || !rigid_motion_stream.is_open() ||
                !penalty_projection_stream.is_open())
            {
                TBOX_ERROR("Unable to open one or more prescribed calibration output files.\n");
            }
            tracking_arclength_stream.precision(12);
            rigid_motion_stream.precision(12);
            penalty_projection_stream.precision(12);
            if (!from_restart)
            {
                tracking_arclength_stream
                    << "time,lab_tracking_2d_rms_over_L,lab_tracking_2d_max_over_L,"
                       "lab_tracking_velocity_2d_rms_over_Lf,"
                       "lab_tracking_velocity_2d_max_over_Lf,"
                       "shape_tracking_2d_rms_over_L,shape_tracking_2d_max_over_L,"
                       "shape_tracking_velocity_2d_rms_over_Lf,"
                       "shape_tracking_velocity_2d_max_over_Lf,"
                       "tracking_midline_rms_over_L,tracking_midline_max_over_L,"
                       "tracking_midline_velocity_rms_over_Lf,"
                       "tracking_midline_velocity_max_over_Lf,"
                       "L_target_rel_error,L_actual_rel_error,lambda_actual_min,lambda_actual_max,"
                       "J_min,area_error\n";
                rigid_motion_stream
                    << "time,x_cm,y_cm,u_cm,v_cm,body_angle,pitch_rate,u_parallel,v_perp\n";
                penalty_projection_stream
                    << "time,Fx_raw,Fy_raw,Mz_raw,Fx_projected,Fy_projected,"
                       "Mz_projected,penalty_power,penalty_force_L1,"
                       "force_projection_residual,torque_projection_residual\n";
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
        if (!from_restart)
        {
            write_diagnostics(mesh,
                              equation_systems,
                              coords_system_name,
                              velocity_system_name,
                              swimmer_data,
                              iteration_num,
                              loop_time);
        }

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

        if (IBTK_MPI::getRank() == 0)
        {
            if (midline_stream.is_open()) midline_stream.close();
            if (tracking_arclength_stream.is_open()) tracking_arclength_stream.close();
            if (rigid_motion_stream.is_open()) rigid_motion_stream.close();
            if (penalty_projection_stream.is_open()) penalty_projection_stream.close();
        }
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

    const BodyFrameFitResult body_frame_fit = fit_body_frames_from_quadrature(
        mesh, *equation_systems, coords_system_name, velocity_system_name, swimmer_data, loop_time);
    const MidlineFrame fitted_actual_frame =
        make_midline_frame_from_rigid_frame(body_frame_fit.actual_frame);
    const MidlineFrame fitted_target_shape_frame =
        make_midline_frame_from_rigid_frame(body_frame_fit.target_shape_frame);

    double volume = 0.0;
    double current_area = 0.0;
    double cm_x_integral = 0.0;
    double cm_y_integral = 0.0;
    double u_x_integral = 0.0;
    double u_y_integral = 0.0;
    double Fx_raw = 0.0;
    double Fy_raw = 0.0;
    double Mz_raw = 0.0;
    double Fx_projected = 0.0;
    double Fy_projected = 0.0;
    double Mz_projected = 0.0;
    double penalty_power = 0.0;
    double penalty_force_L1 = 0.0;
    double J_total_min = std::numeric_limits<double>::max();
    double J_total_max = -std::numeric_limits<double>::max();
    double tracking_2d_lab_error_squared = 0.0;
    double tracking_2d_lab_error_max = 0.0;
    double tracking_2d_lab_velocity_error_squared = 0.0;
    double tracking_2d_lab_velocity_error_max = 0.0;
    double tracking_2d_shape_error_squared = 0.0;
    double tracking_2d_shape_error_max = 0.0;
    double tracking_2d_shape_velocity_error_squared = 0.0;
    double tracking_2d_shape_velocity_error_max = 0.0;

    const int ml_num_output = swimmer_data.midline_num_stations;
    std::vector<double> ml_station_xi;
    std::size_t ml_frame_head_index = 0;
    std::size_t ml_frame_tangent_index = 0;
    std::vector<double> ml_section_weight;
    std::vector<double> ml_section_x;
    std::vector<double> ml_section_y;
    std::vector<double> ml_section_u;
    std::vector<double> ml_section_v;
    std::vector<double> ml_centerline_weight;
    std::vector<double> ml_centerline_x;
    std::vector<double> ml_centerline_y;
    std::vector<double> ml_centerline_u;
    std::vector<double> ml_centerline_v;
    ml_station_xi.reserve(static_cast<std::size_t>(ml_num_output) + 2);
    for (int s = 0; s < ml_num_output; ++s)
        ml_station_xi.push_back(static_cast<double>(s) / (ml_num_output - 1));
    ml_frame_head_index = ml_station_xi.size();
    ml_station_xi.push_back(swimmer_data.midline_body_frame_head_xi);
    ml_frame_tangent_index = ml_station_xi.size();
    ml_station_xi.push_back(swimmer_data.midline_body_frame_tangent_xi);
    const std::size_t n_midline_samples = ml_station_xi.size();
    ml_section_weight.assign(n_midline_samples, 0.0);
    ml_section_x.assign(n_midline_samples, 0.0);
    ml_section_y.assign(n_midline_samples, 0.0);
    ml_section_u.assign(n_midline_samples, 0.0);
    ml_section_v.assign(n_midline_samples, 0.0);
    ml_centerline_weight.assign(n_midline_samples, 0.0);
    ml_centerline_x.assign(n_midline_samples, 0.0);
    ml_centerline_y.assign(n_midline_samples, 0.0);
    ml_centerline_u.assign(n_midline_samples, 0.0);
    ml_centerline_v.assign(n_midline_samples, 0.0);

    boost::multi_array<double, 2> X_node;
    boost::multi_array<double, 2> U_node;
    VectorValue<double> x;
    VectorValue<double> velocity;
    TensorValue<double> FF;
    VectorValue<double> raw_force;
    VectorValue<double> projected_force;
    const libMesh::Point torque_center = swimmer_data.penalty_projection_center;

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

            volume += weight;
            current_area += J_total * weight;
            cm_x_integral += x(0) * weight;
            cm_y_integral += x(1) * weight;
            u_x_integral += velocity(0) * weight;
            u_y_integral += velocity(1) * weight;
            J_total_min = std::min(J_total_min, J_total);
            J_total_max = std::max(J_total_max, J_total);

            const TargetState lab_target = target_state(X, loop_time, swimmer_data);
            const double lab_position_error_squared =
                (lab_target.position(0) - x(0)) * (lab_target.position(0) - x(0)) +
                (lab_target.position(1) - x(1)) * (lab_target.position(1) - x(1));
            const double lab_velocity_error_squared =
                (lab_target.velocity(0) - velocity(0)) * (lab_target.velocity(0) - velocity(0)) +
                (lab_target.velocity(1) - velocity(1)) * (lab_target.velocity(1) - velocity(1));
            tracking_2d_lab_error_squared += lab_position_error_squared * weight;
            tracking_2d_lab_error_max =
                std::max(tracking_2d_lab_error_max, std::sqrt(lab_position_error_squared));
            tracking_2d_lab_velocity_error_squared += lab_velocity_error_squared * weight;
            tracking_2d_lab_velocity_error_max =
                std::max(tracking_2d_lab_velocity_error_max, std::sqrt(lab_velocity_error_squared));

            const TargetState shape_target = target_shape_state(X, loop_time, swimmer_data);
            const std::array<double, 2> actual_shape_position =
                body_frame_coordinates(x, fitted_actual_frame);
            const std::array<double, 2> target_shape_position =
                body_frame_coordinates(shape_target.position, fitted_target_shape_frame);
            const double shape_position_error_squared =
                (target_shape_position[0] - actual_shape_position[0]) *
                    (target_shape_position[0] - actual_shape_position[0]) +
                (target_shape_position[1] - actual_shape_position[1]) *
                    (target_shape_position[1] - actual_shape_position[1]);
            const std::array<double, 2> actual_shape_velocity =
                deformation_velocity_components(x, velocity, body_frame_fit.actual_frame);
            const std::array<double, 2> target_shape_velocity =
                deformation_velocity_components(shape_target.position,
                                                shape_target.velocity,
                                                body_frame_fit.target_shape_frame);
            const double shape_velocity_error_squared =
                (target_shape_velocity[0] - actual_shape_velocity[0]) *
                    (target_shape_velocity[0] - actual_shape_velocity[0]) +
                (target_shape_velocity[1] - actual_shape_velocity[1]) *
                    (target_shape_velocity[1] - actual_shape_velocity[1]);
            tracking_2d_shape_error_squared += shape_position_error_squared * weight;
            tracking_2d_shape_error_max =
                std::max(tracking_2d_shape_error_max, std::sqrt(shape_position_error_squared));
            tracking_2d_shape_velocity_error_squared += shape_velocity_error_squared * weight;
            tracking_2d_shape_velocity_error_max =
                std::max(tracking_2d_shape_velocity_error_max, std::sqrt(shape_velocity_error_squared));

            raw_penalty_force(raw_force, x, X, velocity, loop_time, swimmer_data);
            projected_force = raw_force;
            if (swimmer_data.prescribed_motion_mode == PrescribedMotionMode::FREE_SWIMMING)
            {
                projected_force -= swimmer_data.penalty_projection_translation;
                const VectorValue<double> radius = x - swimmer_data.penalty_projection_center;
                projected_force(0) += swimmer_data.penalty_projection_rotation * radius(1);
                projected_force(1) -= swimmer_data.penalty_projection_rotation * radius(0);
            }
            Fx_raw += raw_force(0) * weight;
            Fy_raw += raw_force(1) * weight;
            Mz_raw += ((x(0) - torque_center(0)) * raw_force(1) -
                       (x(1) - torque_center(1)) * raw_force(0)) *
                      weight;
            Fx_projected += projected_force(0) * weight;
            Fy_projected += projected_force(1) * weight;
            Mz_projected += ((x(0) - torque_center(0)) * projected_force(1) -
                             (x(1) - torque_center(1)) * projected_force(0)) *
                            weight;
            penalty_power += dot2(projected_force, velocity) * weight;
            penalty_force_L1 += std::sqrt(dot2(raw_force, raw_force)) * weight;

            const double ml_eta_abs = std::abs(reference_eta(X, swimmer_data));
            for (std::size_t ml_s = 0; ml_s < ml_station_xi.size(); ++ml_s)
            {
                const double distance = std::abs(xi - ml_station_xi[ml_s]);
                if (distance >= swimmer_data.midline_station_half_width) continue;
                const double sk = (1.0 - distance / swimmer_data.midline_station_half_width) * weight;
                ml_section_weight[ml_s] += sk;
                ml_section_x[ml_s] += x(0) * sk;
                ml_section_y[ml_s] += x(1) * sk;
                ml_section_u[ml_s] += velocity(0) * sk;
                ml_section_v[ml_s] += velocity(1) * sk;
                if (ml_eta_abs < swimmer_data.midline_centerline_half_thickness)
                {
                    const double ek =
                        (1.0 - ml_eta_abs / swimmer_data.midline_centerline_half_thickness) * sk;
                    ml_centerline_weight[ml_s] += ek;
                    ml_centerline_x[ml_s] += x(0) * ek;
                    ml_centerline_y[ml_s] += x(1) * ek;
                    ml_centerline_u[ml_s] += velocity(0) * ek;
                    ml_centerline_v[ml_s] += velocity(1) * ek;
                }
            }
        }
    }

    const std::array<double*, 18> sums = { &volume,
                                           &current_area,
                                           &cm_x_integral,
                                           &cm_y_integral,
                                           &u_x_integral,
                                           &u_y_integral,
                                           &Fx_raw,
                                           &Fy_raw,
                                           &Mz_raw,
                                           &Fx_projected,
                                           &Fy_projected,
                                           &Mz_projected,
                                           &penalty_power,
                                           &penalty_force_L1,
                                           &tracking_2d_lab_error_squared,
                                           &tracking_2d_lab_velocity_error_squared,
                                           &tracking_2d_shape_error_squared,
                                           &tracking_2d_shape_velocity_error_squared };
    for (double* value : sums) IBTK_MPI::sumReduction(value, 1);
    IBTK_MPI::minReduction(&J_total_min, 1);
    IBTK_MPI::maxReduction(&J_total_max, 1);
    IBTK_MPI::maxReduction(&tracking_2d_lab_error_max, 1);
    IBTK_MPI::maxReduction(&tracking_2d_lab_velocity_error_max, 1);
    IBTK_MPI::maxReduction(&tracking_2d_shape_error_max, 1);
    IBTK_MPI::maxReduction(&tracking_2d_shape_velocity_error_max, 1);
    IBTK_MPI::sumReduction(ml_section_weight.data(), ml_section_weight.size());
    IBTK_MPI::sumReduction(ml_section_x.data(), ml_section_x.size());
    IBTK_MPI::sumReduction(ml_section_y.data(), ml_section_y.size());
    IBTK_MPI::sumReduction(ml_section_u.data(), ml_section_u.size());
    IBTK_MPI::sumReduction(ml_section_v.data(), ml_section_v.size());
    IBTK_MPI::sumReduction(ml_centerline_weight.data(), ml_centerline_weight.size());
    IBTK_MPI::sumReduction(ml_centerline_x.data(), ml_centerline_x.size());
    IBTK_MPI::sumReduction(ml_centerline_y.data(), ml_centerline_y.size());
    IBTK_MPI::sumReduction(ml_centerline_u.data(), ml_centerline_u.size());
    IBTK_MPI::sumReduction(ml_centerline_v.data(), ml_centerline_v.size());

    std::vector<MidlineSample> ml_samples;
    const int ml_fallback_count = assemble_midline_samples(
        ml_section_weight, ml_section_x, ml_section_y, ml_section_u, ml_section_v,
        ml_centerline_weight, ml_centerline_x, ml_centerline_y,
        ml_centerline_u, ml_centerline_v,
        ml_station_xi, iteration_num, loop_time, swimmer_data, ml_samples);

    if (volume <= 0.0) TBOX_ERROR("Cannot compute diagnostics on a zero-volume mesh.\n");
    libMesh::Point center_of_mass;
    center_of_mass(0) = cm_x_integral / volume;
    center_of_mass(1) = cm_y_integral / volume;
    VectorValue<double> center_velocity;
    center_velocity(0) = u_x_integral / volume;
    center_velocity(1) = u_y_integral / volume;
    MidlineFrame actual_frame = make_reference_midline_frame(swimmer_data);
    RigidTargetFrame actual_tracking_frame = make_reference_target_frame(swimmer_data);
    if (swimmer_data.midline_use_body_frame)
    {
        actual_frame = fitted_actual_frame;
        actual_tracking_frame = body_frame_fit.actual_frame;
    }
    const double body_angle = std::atan2(dot2(body_frame_fit.actual_frame.tangent,
                                             swimmer_data.geometry.normal),
                                         dot2(body_frame_fit.actual_frame.tangent,
                                              swimmer_data.geometry.tangent));
    const double actual_frame_pitch_rate = body_frame_fit.actual_frame.angular_rate;

    std::vector<libMesh::Point> actual_points(ml_num_output);
    std::vector<libMesh::Point> target_points(ml_num_output);
    std::vector<TargetState> target_samples(ml_station_xi.size());
    std::vector<TargetState> target_shape_samples(ml_station_xi.size());
    for (std::size_t station = 0; station < ml_station_xi.size(); ++station)
    {
        const libMesh::Point X = reference_centerline_point(ml_station_xi[station], swimmer_data);
        target_samples[station] = target_state(X, loop_time, swimmer_data);
        target_shape_samples[station] = target_shape_state(X, loop_time, swimmer_data);
    }
    MidlineFrame target_frame = make_reference_midline_frame(swimmer_data);
    RigidTargetFrame target_tracking_frame = make_reference_target_frame(swimmer_data);
    if (swimmer_data.midline_use_body_frame)
    {
        target_frame = fitted_target_shape_frame;
        target_tracking_frame = body_frame_fit.target_shape_frame;
    }
    for (int station = 0; station < ml_num_output; ++station)
    {
        const std::size_t index = static_cast<std::size_t>(station);
        actual_points[index] = ml_samples[index].position;
        target_points[index] = target_samples[index].position;
    }

    double L_target = 0.0;
    double L_actual = 0.0;
    double lambda_actual_min = std::numeric_limits<double>::max();
    double lambda_actual_max = -std::numeric_limits<double>::max();
    for (int station = 1; station < ml_num_output; ++station)
    {
        const std::size_t i = static_cast<std::size_t>(station);
        const double ds_ref = (ml_station_xi[i] - ml_station_xi[i - 1]) * swimmer_data.geometry.length;
        const double ds_target = (target_points[i] - target_points[i - 1]).norm();
        const double ds_actual = (actual_points[i] - actual_points[i - 1]).norm();
        L_target += ds_target;
        L_actual += ds_actual;
        if (ds_ref > 0.0)
        {
            const double lambda_actual = ds_actual / ds_ref;
            lambda_actual_min = std::min(lambda_actual_min, lambda_actual);
            lambda_actual_max = std::max(lambda_actual_max, lambda_actual);
        }
    }

    double tracking_error_squared = 0.0;
    double tracking_error_max = 0.0;
    double tracking_velocity_error_squared = 0.0;
    double tracking_velocity_error_max = 0.0;
    for (int station = 0; station < ml_num_output; ++station)
    {
        const std::size_t index = static_cast<std::size_t>(station);
        double ex = 0.0;
        double ey = 0.0;
        double eu = 0.0;
        double ev = 0.0;
        if (swimmer_data.midline_use_body_frame)
        {
            const std::array<double, 2> actual_body =
                body_frame_coordinates(actual_points[index], actual_frame);
            const std::array<double, 2> target_body =
                body_frame_coordinates(target_shape_samples[index].position, target_frame);
            ex = target_body[0] - actual_body[0];
            ey = target_body[1] - actual_body[1];

            const std::array<double, 2> actual_velocity_body =
                deformation_velocity_components(actual_points[index],
                                                ml_samples[index].velocity,
                                                actual_tracking_frame);
            const std::array<double, 2> target_velocity_body =
                deformation_velocity_components(target_shape_samples[index].position,
                                                target_shape_samples[index].velocity,
                                                target_tracking_frame);
            eu = target_velocity_body[0] - actual_velocity_body[0];
            ev = target_velocity_body[1] - actual_velocity_body[1];
        }
        else
        {
            ex = target_points[index](0) - actual_points[index](0);
            ey = target_points[index](1) - actual_points[index](1);
            eu = target_samples[index].velocity(0) - ml_samples[index].velocity(0);
            ev = target_samples[index].velocity(1) - ml_samples[index].velocity(1);
        }
        const double error_squared = ex * ex + ey * ey;
        tracking_error_squared += error_squared;
        tracking_error_max = std::max(tracking_error_max, std::sqrt(error_squared));
        const double velocity_error_squared = eu * eu + ev * ev;
        tracking_velocity_error_squared += velocity_error_squared;
        tracking_velocity_error_max =
            std::max(tracking_velocity_error_max, std::sqrt(velocity_error_squared));
    }
    const double tracking_error_rms =
        std::sqrt(tracking_error_squared / static_cast<double>(ml_num_output));
    const double tracking_velocity_error_rms =
        std::sqrt(tracking_velocity_error_squared / static_cast<double>(ml_num_output));
    const double characteristic_velocity = swimmer_data.geometry.length * swimmer_data.frequency;
    const double tracking_2d_lab_error_rms =
        std::sqrt(tracking_2d_lab_error_squared / volume);
    const double tracking_2d_lab_velocity_error_rms =
        std::sqrt(tracking_2d_lab_velocity_error_squared / volume);
    const double tracking_2d_shape_error_rms =
        std::sqrt(tracking_2d_shape_error_squared / volume);
    const double tracking_2d_shape_velocity_error_rms =
        std::sqrt(tracking_2d_shape_velocity_error_squared / volume);
    const double L_target_rel_error = (L_target - swimmer_data.geometry.length) / swimmer_data.geometry.length;
    const double L_actual_rel_error = (L_actual - swimmer_data.geometry.length) / swimmer_data.geometry.length;
    const double area_error = (current_area - volume) / volume;
    const double u_parallel = -dot2(center_velocity, actual_frame.tangent);
    const double v_perp = dot2(center_velocity, actual_frame.normal);
    const double projection_epsilon =
        100.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, penalty_force_L1);
    const double force_projection_residual =
        std::sqrt(Fx_projected * Fx_projected + Fy_projected * Fy_projected) /
        (penalty_force_L1 + projection_epsilon);
    const double torque_projection_residual =
        std::abs(Mz_projected) /
        (swimmer_data.geometry.length * penalty_force_L1 + projection_epsilon);

    if (IBTK_MPI::getRank() == 0)
    {
        tracking_arclength_stream.setf(std::ios::scientific);
        rigid_motion_stream.setf(std::ios::scientific);
        penalty_projection_stream.setf(std::ios::scientific);
        tracking_arclength_stream
            << loop_time << ","
            << tracking_2d_lab_error_rms / swimmer_data.geometry.length << ","
            << tracking_2d_lab_error_max / swimmer_data.geometry.length << ","
            << tracking_2d_lab_velocity_error_rms / characteristic_velocity << ","
            << tracking_2d_lab_velocity_error_max / characteristic_velocity << ","
            << tracking_2d_shape_error_rms / swimmer_data.geometry.length << ","
            << tracking_2d_shape_error_max / swimmer_data.geometry.length << ","
            << tracking_2d_shape_velocity_error_rms / characteristic_velocity << ","
            << tracking_2d_shape_velocity_error_max / characteristic_velocity << ","
            << tracking_error_rms / swimmer_data.geometry.length << ","
            << tracking_error_max / swimmer_data.geometry.length << ","
            << tracking_velocity_error_rms / characteristic_velocity << ","
            << tracking_velocity_error_max / characteristic_velocity << ","
            << L_target_rel_error << "," << L_actual_rel_error << ","
            << lambda_actual_min << "," << lambda_actual_max << ","
            << J_total_min << "," << area_error << "\n";
        rigid_motion_stream << loop_time << "," << center_of_mass(0) << ","
                            << center_of_mass(1) << "," << center_velocity(0) << ","
                            << center_velocity(1) << "," << body_angle << ","
                            << actual_frame_pitch_rate << "," << u_parallel << ","
                            << v_perp << "\n";
        penalty_projection_stream << loop_time << "," << Fx_raw << "," << Fy_raw << ","
                                  << Mz_raw << "," << Fx_projected << "," << Fy_projected << ","
                                  << Mz_projected << "," << penalty_power << ","
                                  << penalty_force_L1 << "," << force_projection_residual << ","
                                  << torque_projection_residual << "\n";
        tracking_arclength_stream.flush();
        rigid_motion_stream.flush();
        penalty_projection_stream.flush();
    }

    write_midline_output(ml_samples, ml_station_xi,
                         ml_frame_head_index, ml_frame_tangent_index,
                         ml_num_output, ml_fallback_count,
                         swimmer_data, iteration_num, loop_time);

    if (!std::isfinite(J_total_min) || !std::isfinite(J_total_max) || J_total_min <= 0.0)
    {
        TBOX_ERROR("Invalid structural mesh Jacobian at step "
                   << iteration_num << ", time = " << loop_time << ": J range = [" << J_total_min << ","
                   << J_total_max << "]. A finite positive Jacobian is required.\n");
    }
    if (J_total_min < swimmer_data.min_allowed_J || J_total_max > swimmer_data.max_allowed_J)
    {
        if (swimmer_data.abort_on_mesh_quality_limit)
        {
            TBOX_ERROR("Structural mesh quality limit exceeded at step "
                       << iteration_num << ", time = " << loop_time << ": J range = [" << J_total_min << ","
                       << J_total_max << "], allowed range = [" << swimmer_data.min_allowed_J << ","
                       << swimmer_data.max_allowed_J << "].\n");
        }
        if (IBTK_MPI::getRank() == 0)
        {
            pout << "WARNING: structural mesh quality threshold exceeded at step "
                 << iteration_num << ", time = " << loop_time << ": J range = [" << J_total_min << ","
                 << J_total_max << "], warning range = [" << swimmer_data.min_allowed_J << ","
                 << swimmer_data.max_allowed_J << "].\n";
        }
    }
    const bool use_lab_tracking_limit =
        swimmer_data.prescribed_motion_mode == PrescribedMotionMode::TETHERED;
    const double tracking_limit_error =
        use_lab_tracking_limit ? tracking_2d_lab_error_max : tracking_2d_shape_error_max;
    const double tracking_limit_velocity_error =
        use_lab_tracking_limit ? tracking_2d_lab_velocity_error_max :
                                 tracking_2d_shape_velocity_error_max;
    const std::string tracking_limit_name =
        use_lab_tracking_limit ? "lab-frame 2D" : "body-frame shape 2D";
    if (tracking_limit_error / swimmer_data.geometry.length >
        swimmer_data.max_tracking_error_over_L)
    {
        TBOX_ERROR("Penalty tracking error limit exceeded at step "
                   << iteration_num << ", time = " << loop_time << ": "
                   << tracking_limit_name << " max error/L = "
                   << tracking_limit_error / swimmer_data.geometry.length
                   << ", allowed maximum = " << swimmer_data.max_tracking_error_over_L << ".\n");
    }
    if (tracking_limit_velocity_error / characteristic_velocity >
        swimmer_data.max_tracking_velocity_error_over_Lf)
    {
        TBOX_ERROR("Penalty velocity tracking limit exceeded at step "
                   << iteration_num << ", time = " << loop_time << ": "
                   << tracking_limit_name << " max |U_target-U|/(L*f) = "
                   << tracking_limit_velocity_error / characteristic_velocity
                   << ", allowed maximum = " << swimmer_data.max_tracking_velocity_error_over_Lf
                   << ".\n");
    }
}

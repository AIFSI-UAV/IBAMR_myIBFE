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

enum class ActiveCurvatureMode
{
    GAIT_MATCHED,
    STATIC_UNIFORM
};

enum class ActiveTransverseMode
{
    NONE,
    AREA_PRESERVING
};

ActiveCurvatureMode
parse_active_curvature_mode(const std::string& value)
{
    if (value == "GAIT_MATCHED") return ActiveCurvatureMode::GAIT_MATCHED;
    if (value == "STATIC_UNIFORM") return ActiveCurvatureMode::STATIC_UNIFORM;
    TBOX_ERROR("ACTIVE_CURVATURE_MODE must be GAIT_MATCHED or STATIC_UNIFORM.\n");
    return ActiveCurvatureMode::GAIT_MATCHED;
}

ActiveTransverseMode
parse_active_transverse_mode(const std::string& value)
{
    if (value == "NONE") return ActiveTransverseMode::NONE;
    if (value == "AREA_PRESERVING") return ActiveTransverseMode::AREA_PRESERVING;
    TBOX_ERROR("ACTIVE_TRANSVERSE_MODE must be NONE or AREA_PRESERVING.\n");
    return ActiveTransverseMode::AREA_PRESERVING;
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

struct ActiveMetricState
{
    TensorValue<double> gradient;
    TensorValue<double> inverse;
    double curvature = 0.0;
    double axial_stretch = 1.0;
    double J_active = 1.0;
    double longitudinal_shear = 0.0;
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

    ActiveCurvatureMode active_curvature_mode;
    ActiveTransverseMode active_transverse_mode;
    double active_curvature_scale;
    double static_active_curvature;
    double active_test_tether_fraction;
    double active_test_tether_stiffness;
    double active_test_tether_damping;

    bool run_material_self_checks;
    double material_self_check_tolerance;
    double diagnostic_station_half_width;
    double diagnostic_centerline_half_thickness;
    double min_allowed_J;
    double max_allowed_J;
    bool abort_on_mesh_quality_limit;
    double max_tracking_error_over_L;
    double max_tracking_velocity_error_over_Lf;
    bool write_midline_data;
    std::string midline_filename;
    int midline_num_stations;
    double midline_station_half_width;
    double midline_centerline_half_thickness;
    bool midline_use_body_frame;
    double midline_body_frame_head_xi;
    double midline_body_frame_tangent_xi;

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
          active_curvature_mode(
              parse_active_curvature_mode(input_db->getString("ACTIVE_CURVATURE_MODE"))),
          active_transverse_mode(
              parse_active_transverse_mode(input_db->getString("ACTIVE_TRANSVERSE_MODE"))),
          active_curvature_scale(input_db->getDouble("ACTIVE_CURVATURE_SCALE")),
          static_active_curvature(input_db->getDouble("STATIC_ACTIVE_CURVATURE_TIMES_L") /
                                  reference_geometry.length),
          active_test_tether_fraction(input_db->getDouble("ACTIVE_TEST_TETHER_FRACTION")),
          active_test_tether_stiffness(input_db->getDouble("ACTIVE_TEST_TETHER_STIFFNESS")),
          active_test_tether_damping(input_db->getDouble("ACTIVE_TEST_TETHER_DAMPING")),
          run_material_self_checks(input_db->getBool("RUN_MATERIAL_SELF_CHECKS")),
          material_self_check_tolerance(input_db->getDouble("MATERIAL_SELF_CHECK_TOLERANCE")),
          diagnostic_station_half_width(input_db->getDouble("DIAGNOSTIC_STATION_HALF_WIDTH")),
          diagnostic_centerline_half_thickness(
              input_db->getDouble("DIAGNOSTIC_CENTERLINE_HALF_THICKNESS")),
          min_allowed_J(input_db->getDouble("MIN_ALLOWED_J")),
          max_allowed_J(input_db->getDouble("MAX_ALLOWED_J")),
          abort_on_mesh_quality_limit(input_db->getBool("ABORT_ON_MESH_QUALITY_LIMIT")),
          max_tracking_error_over_L(input_db->getDouble("MAX_TRACKING_ERROR_OVER_L")),
          max_tracking_velocity_error_over_Lf(
              input_db->getDouble("MAX_TRACKING_VELOCITY_ERROR_OVER_LF")),
          write_midline_data(input_db->getBoolWithDefault("WRITE_MIDLINE_DATA", true)),
          midline_filename(
              input_db->getStringWithDefault("MIDLINE_FILENAME", "fish_midline_history.csv")),
          midline_num_stations(input_db->getIntegerWithDefault("MIDLINE_NUM_STATIONS", 151)),
          midline_station_half_width(
              input_db->getDoubleWithDefault("MIDLINE_STATION_HALF_WIDTH", 0.008)),
          midline_centerline_half_thickness(
              input_db->getDoubleWithDefault("MIDLINE_CENTERLINE_HALF_THICKNESS", 0.003)),
          midline_use_body_frame(input_db->getBoolWithDefault("MIDLINE_USE_BODY_FRAME", true)),
          midline_body_frame_head_xi(
              input_db->getDoubleWithDefault("MIDLINE_BODY_FRAME_HEAD_XI", 0.02)),
          midline_body_frame_tangent_xi(
              input_db->getDoubleWithDefault("MIDLINE_BODY_FRAME_TANGENT_XI", 0.10))
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
        {
            const bool p_is_one = std::abs(prescribed_envelope_power - 1.0) < 1.0e-12;
            const bool p_is_two = std::abs(prescribed_envelope_power - 2.0) < 1.0e-12;
            if (!(p_is_one || p_is_two || prescribed_envelope_power >= 3.0))
            {
                TBOX_ERROR("PRESCRIBED_ENVELOPE_POWER must be 1, 2, or >= 3 "
                           "because kappa_s requires a finite third envelope derivative at s=0.\n");
            }
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
        if (active_curvature_scale <= 0.0)
        {
            TBOX_ERROR("ACTIVE_CURVATURE_SCALE must be positive.\n");
        }
        if (!std::isfinite(static_active_curvature))
        {
            TBOX_ERROR("STATIC_ACTIVE_CURVATURE_TIMES_L must be finite.\n");
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
        if (diagnostic_centerline_half_thickness <= 0.0)
        {
            TBOX_ERROR("DIAGNOSTIC_CENTERLINE_HALF_THICKNESS must be positive.\n");
        }
        if (!(0.0 < min_allowed_J && min_allowed_J < 1.0 && 1.0 < max_allowed_J))
        {
            TBOX_ERROR("Require 0 < MIN_ALLOWED_J < 1 < MAX_ALLOWED_J.\n");
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

struct MidlineSample
{
    double xi = 0.0;
    libMesh::Point position;
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

double
prescribed_amplitude_third_derivative(const double s, const SwimmerData& data)
{
    const double p = data.prescribed_envelope_power;
    if (std::abs(p - 1.0) < 1.0e-12 || std::abs(p - 2.0) < 1.0e-12) return 0.0;
    const double xi = unit_coordinate(s, data);
    const double delta = data.prescribed_tail_amplitude - data.prescribed_head_amplitude;
    const double L = data.geometry.length;
    const double coefficient = delta * p * (p - 1.0) * (p - 2.0) / (L * L * L);
    if (std::abs(p - 3.0) < 1.0e-12) return coefficient;
    if (xi <= 0.0) return 0.0;
    return coefficient * std::pow(xi, p - 3.0);
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
active_curvature_at_s(const double s, const double time, const SwimmerData& data)
{
    double ramp_value = 0.0;
    double ramp_derivative = 0.0;
    ramp(time, data, ramp_value, ramp_derivative);
    if (data.active_curvature_mode == ActiveCurvatureMode::STATIC_UNIFORM)
    {
        return data.active_curvature_scale * ramp_value * data.static_active_curvature;
    }

    const double phase = wave_number(data) * s - angular_frequency(data) * time + data.phase0;
    double amplitude = 0.0;
    double amplitude_derivative = 0.0;
    prescribed_amplitude(s, data, amplitude, amplitude_derivative);
    const double amplitude_second_derivative = prescribed_amplitude_second_derivative(s, data);
    const double k = wave_number(data);
    const double spatial_wave =
        amplitude_derivative * std::sin(phase) + amplitude * k * std::cos(phase);
    const double spatial_wave_derivative =
        amplitude_second_derivative * std::sin(phase) +
        2.0 * amplitude_derivative * k * std::cos(phase) -
        amplitude * k * k * std::sin(phase);
    const double slope = ramp_value * spatial_wave;
    return data.active_curvature_scale * ramp_value * spatial_wave_derivative /
           (1.0 + slope * slope);
}

double
maximum_active_curvature_magnitude(const SwimmerData& data)
{
    constexpr unsigned int num_space_samples = 512;
    constexpr unsigned int num_time_samples = 512;
    const double period = 1.0 / data.frequency;
    const double end_time = data.ramp_time + period;
    double maximum = 0.0;
    for (unsigned int time_sample = 0; time_sample <= num_time_samples; ++time_sample)
    {
        const double time =
            end_time * static_cast<double>(time_sample) / static_cast<double>(num_time_samples);
        for (unsigned int space_sample = 0; space_sample <= num_space_samples; ++space_sample)
        {
            const double s = data.geometry.length * static_cast<double>(space_sample) /
                             static_cast<double>(num_space_samples);
            maximum = std::max(maximum, std::abs(active_curvature_at_s(s, time, data)));
        }
    }
    return maximum;
}

double
active_curvature(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    return active_curvature_at_s(reference_s(X, data), time, data);
}

double
active_angle_at_s(const double s, const double time, const SwimmerData& data)
{
    if (data.active_curvature_mode == ActiveCurvatureMode::STATIC_UNIFORM)
    {
        return active_curvature_at_s(s, time, data) * s;
    }

    double theta = 0.0;
    double theta_time = 0.0;
    double theta_head = 0.0;
    double theta_head_time = 0.0;
    target_angle(s, time, data, theta, theta_time);
    target_angle(0.0, time, data, theta_head, theta_head_time);
    return data.active_curvature_scale * (theta - theta_head);
}

double
active_curvature_spatial_derivative(const double s,
                                    const double time,
                                    const SwimmerData& data)
{
    if (data.active_curvature_mode == ActiveCurvatureMode::STATIC_UNIFORM) return 0.0;

    double ramp_value = 0.0;
    double ramp_derivative = 0.0;
    ramp(time, data, ramp_value, ramp_derivative);

    const double k = wave_number(data);
    const double phase = k * s - angular_frequency(data) * time + data.phase0;
    const double sin_phase = std::sin(phase);
    const double cos_phase = std::cos(phase);

    double amplitude = 0.0;
    double amplitude_s = 0.0;
    prescribed_amplitude(s, data, amplitude, amplitude_s);
    const double amplitude_ss = prescribed_amplitude_second_derivative(s, data);
    const double amplitude_sss = prescribed_amplitude_third_derivative(s, data);

    const double h_s = amplitude_s * sin_phase + amplitude * k * cos_phase;
    const double h_ss = (amplitude_ss - k * k * amplitude) * sin_phase +
                        2.0 * k * amplitude_s * cos_phase;
    const double h_sss = (amplitude_sss - 3.0 * k * k * amplitude_s) * sin_phase +
                         k * (3.0 * amplitude_ss - k * k * amplitude) * cos_phase;

    const double slope = ramp_value * h_s;
    const double denom = 1.0 + slope * slope;
    return data.active_curvature_scale * ramp_value *
           (h_sss * denom - 2.0 * slope * ramp_value * h_ss * h_ss) /
           (denom * denom);
}

libMesh::Point
active_preferred_centerline(const double s, const double time, const SwimmerData& data)
{
    const unsigned int num_intervals =
        std::max(16U, static_cast<unsigned int>(std::ceil(256.0 * s / data.geometry.length)));
    const double ds = s / static_cast<double>(num_intervals);
    double local_x = 0.0;
    double local_y = 0.0;
    double theta = 0.0;
    const auto curvature = [&](const double coordinate) {
        return active_curvature_at_s(coordinate, time, data);
    };
    for (unsigned int interval = 0; interval < num_intervals; ++interval)
    {
        const double s_mid = (static_cast<double>(interval) + 0.5) * ds;
        const double kappa_mid = curvature(s_mid);
        const double theta_mid = theta + 0.5 * ds * kappa_mid;
        local_x += ds * std::cos(theta_mid);
        local_y += ds * std::sin(theta_mid);
        theta += ds * kappa_mid;
    }

    libMesh::Point point = data.geometry.head;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        point(d) += local_x * data.geometry.tangent(d) + local_y * data.geometry.normal(d);
    }
    return point;
}

bool
uses_active_strain(const SwimmerData& data)
{
    return data.stage == SwimmerStage::ACTIVE_STRAIN ||
           data.stage == SwimmerStage::ACTIVE_STRAIN_TETHERED_TEST;
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

ActiveMetricState
identity_active_metric()
{
    ActiveMetricState metric;
    metric.gradient.zero();
    metric.inverse.zero();
    for (unsigned int d = 0; d < 3; ++d)
    {
        metric.gradient(d, d) = 1.0;
        metric.inverse(d, d) = 1.0;
    }
    return metric;
}

ActiveMetricState
active_metric_from_kinematics(const double eta,
                              const double curvature,
                              const double curvature_s,
                              const double angle,
                              const SwimmerData& data)
{
    ActiveMetricState metric;
    metric.curvature = curvature;

    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const VectorValue<double> active_tangent =
        cosine * data.geometry.tangent + sine * data.geometry.normal;
    const VectorValue<double> active_normal =
        -sine * data.geometry.tangent + cosine * data.geometry.normal;

    double normal_coordinate = eta;
    double transverse_stretch = 1.0;
    if (data.active_transverse_mode == ActiveTransverseMode::AREA_PRESERVING)
    {
        const double discriminant = 1.0 - 2.0 * eta * curvature;
        if (!(discriminant > 0.0) || !std::isfinite(discriminant))
        {
            TBOX_ERROR("Compatible area-preserving active metric requires 1-2*eta*kappa > 0, "
                       "but obtained "
                       << discriminant << ".\n");
        }
        metric.axial_stretch = std::sqrt(discriminant);
        normal_coordinate = 2.0 * eta / (1.0 + metric.axial_stretch);
        transverse_stretch = 1.0 / metric.axial_stretch;
        metric.longitudinal_shear =
            0.5 * normal_coordinate * normal_coordinate * curvature_s /
            metric.axial_stretch;
        metric.J_active = 1.0;
    }
    else
    {
        metric.axial_stretch = 1.0 - eta * curvature;
        metric.J_active = metric.axial_stretch;
    }
    if (!(metric.axial_stretch > 0.0) || !std::isfinite(metric.axial_stretch))
    {
        TBOX_ERROR("Active bending metric produced a nonpositive or nonfinite axial stretch: "
                   << metric.axial_stretch << ".\n");
    }

    const VectorValue<double> longitudinal_column =
        metric.axial_stretch * active_tangent +
        metric.longitudinal_shear * active_normal;
    const VectorValue<double> transverse_column =
        transverse_stretch * active_normal;
    metric.gradient =
        dyad(longitudinal_column, data.geometry.tangent) +
        dyad(transverse_column, data.geometry.normal);
    metric.gradient(2, 2) = 1.0;
    tensor_inverse(metric.inverse, metric.gradient, NDIM);

    const double determinant =
        metric.gradient(0, 0) * metric.gradient(1, 1) -
        metric.gradient(0, 1) * metric.gradient(1, 0);
    const double determinant_tolerance =
        100.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(metric.J_active));
    if (!std::isfinite(determinant) ||
        std::abs(determinant - metric.J_active) > determinant_tolerance)
    {
        TBOX_ERROR("Active metric determinant mismatch: det(Fa)="
                   << determinant << ", expected Ja=" << metric.J_active << ".\n");
    }
    return metric;
}

ActiveMetricState
active_metric_state(const libMesh::Point& X, const double time, const SwimmerData& data)
{
    if (!uses_active_strain(data)) return identity_active_metric();

    const double s = reference_s(X, data);
    const double eta = reference_eta(X, data);
    return active_metric_from_kinematics(eta,
                                         active_curvature_at_s(s, time, data),
                                         active_curvature_spatial_derivative(s, time, data),
                                         active_angle_at_s(s, time, data),
                                         data);
}

MaterialResponse
evaluate_material(const TensorValue<double>& FF,
                  const ActiveMetricState& active_metric,
                  const SwimmerData& data,
                  const libMesh::Point* X,
                  const double time)
{
    const TensorValue<double> Fe = multiply2(FF, active_metric.inverse);

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

    response.matrix =
        pull_back_active_stress(Pe_matrix, active_metric.inverse, active_metric.J_active);
    response.fiber =
        pull_back_active_stress(Pe_fiber, active_metric.inverse, active_metric.J_active);
    response.shear =
        pull_back_active_stress(Pe_shear, active_metric.inverse, active_metric.J_active);
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
    const ActiveMetricState active_metric = active_metric_state(X, time, data);
    PP = evaluate_material(FF, active_metric, data, &X, time).total;
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
    const double characteristic_velocity = data.geometry.length * data.frequency;
    const double velocity_error_over_Lf =
        std::sqrt(dot2(velocity_error, velocity_error)) / characteristic_velocity;
    if (velocity_error_over_Lf > data.max_tracking_velocity_error_over_Lf)
    {
        TBOX_ERROR("Penalty velocity tracking limit exceeded at time = "
                   << time << ": |U_target-U|/(L*f) = " << velocity_error_over_Lf
                   << ", allowed maximum = " << data.max_tracking_velocity_error_over_Lf << ".\n");
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
    const ActiveMetricState identity_metric = identity_active_metric();
    const MaterialResponse passive =
        evaluate_material(identity, identity_metric, data, nullptr, 0.0);
    const double passive_residual = frobenius_norm(passive.total);

    const double check_time = data.ramp_time + 0.25 / data.frequency;
    const double check_s = 0.8 * data.geometry.length;
    const double eta = 0.5 * (data.geometry.eta_min + data.geometry.eta_max) +
                       0.45 * (data.geometry.eta_max - data.geometry.eta_min);
    const ActiveMetricState active_metric =
        active_metric_from_kinematics(eta,
                                      active_curvature_at_s(check_s, check_time, data),
                                      active_curvature_spatial_derivative(check_s, check_time, data),
                                      active_angle_at_s(check_s, check_time, data),
                                      data);
    const MaterialResponse active_stress_free =
        evaluate_material(active_metric.gradient, active_metric, data, nullptr, check_time);
    const double active_residual = frobenius_norm(active_stress_free.total);

    const double eta_test = 0.4 * std::max(std::abs(data.geometry.eta_min),
                                           std::abs(data.geometry.eta_max));
    const double curvature_test = 0.5 * maximum_active_curvature_magnitude(data);
    const ActiveMetricState positive_eta_metric =
        active_metric_from_kinematics(eta_test, curvature_test, 0.0, 0.0, data);
    const ActiveMetricState negative_eta_metric =
        active_metric_from_kinematics(-eta_test, curvature_test, 0.0, 0.0, data);
    const MaterialResponse positive_eta_stress =
        evaluate_material(identity, positive_eta_metric, data, nullptr, check_time);
    const MaterialResponse negative_eta_stress =
        evaluate_material(identity, negative_eta_metric, data, nullptr, check_time);
    const auto axial_stress = [&](const TensorValue<double>& stress) {
        return dot2(data.geometry.tangent, stress * data.geometry.tangent);
    };
    const double active_moment_sign_check =
        eta_test * (axial_stress(positive_eta_stress.total) -
                    axial_stress(negative_eta_stress.total));

    pout << "Material self-check: ||P(F=I,Fa=I)|| = " << passive_residual
         << ", ||P(F=Fa)|| = " << active_residual
         << ", positive-curvature straight-state moment = " << active_moment_sign_check << "\n";
    if (!std::isfinite(passive_residual) || !std::isfinite(active_residual) ||
        passive_residual > data.material_self_check_tolerance ||
        active_residual > data.material_self_check_tolerance)
    {
        TBOX_ERROR("Active-strain material self-check failed. Tolerance = "
                   << data.material_self_check_tolerance << "\n");
    }
    if (uses_active_strain(data) &&
        (!std::isfinite(active_moment_sign_check) || active_moment_sign_check <= 0.0))
    {
        TBOX_ERROR("Active-curvature sign self-check failed: positive diagnostic curvature must "
                   "produce a positive straight-state stress moment.\n");
    }
}
} // namespace ModelData
using namespace ModelData;

static std::ofstream diagnostics_stream;
static std::ofstream midline_stream;

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
                   const libMesh::Point& tangent_point,
                   const SwimmerData& swimmer_data)
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
    (void)swimmer_data;
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

std::array<double, 2>
body_frame_coordinates(const libMesh::Point& point, const MidlineFrame& frame)
{
    const VectorValue<double> displacement = point - frame.origin;
    return { dot2(displacement, frame.tangent), dot2(displacement, frame.normal) };
}

int
assemble_midline_samples(const std::vector<double>& section_weight,
                         const std::vector<double>& section_x,
                         const std::vector<double>& section_y,
                         const std::vector<double>& centerline_weight,
                         const std::vector<double>& centerline_x,
                         const std::vector<double>& centerline_y,
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
            samples[station].source = "ETA0";
        }
        else if (section_weight[station] > 0.0)
        {
            samples[station].position(0) = section_x[station] / section_weight[station];
            samples[station].position(1) = section_y[station] / section_weight[station];
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
                                          samples[frame_tangent_index].position,
                                          swimmer_data);
    }
    const MidlineFrame& comparison_frame = actual_frame;

    const bool write_target = swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED;
    const bool write_preferred = uses_active_strain(swimmer_data);
    std::vector<libMesh::Point> target_points(station_xi.size());
    std::vector<libMesh::Point> preferred_points(station_xi.size());
    for (std::size_t station = 0; station < station_xi.size(); ++station)
    {
        const libMesh::Point X = reference_centerline_point(station_xi[station], swimmer_data);
        if (write_target)
            target_points[station] = target_state(X, loop_time, swimmer_data).position;
        if (write_preferred)
            preferred_points[station] = active_preferred_centerline(
                station_xi[station] * swimmer_data.geometry.length, loop_time, swimmer_data);
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

    const double L = swimmer_data.geometry.length;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    midline_stream.setf(std::ios::scientific);
    midline_stream.precision(12);
    for (int station = 0; station < num_output_stations; ++station)
    {
        const std::size_t index = static_cast<std::size_t>(station);
        const std::array<double, 2> actual_body =
            body_frame_coordinates(samples[index].position, comparison_frame);
        double target_x_body = nan;
        double target_y_body = nan;
        double preferred_x_body = nan;
        double preferred_y_body = nan;
        if (write_target)
        {
            const std::array<double, 2> target_body =
                body_frame_coordinates(target_points[index], comparison_frame);
            target_x_body = target_body[0] / L;
            target_y_body = target_body[1] / L;
        }
        if (write_preferred)
        {
            const std::array<double, 2> preferred_body =
                body_frame_coordinates(preferred_points[index], comparison_frame);
            preferred_x_body = preferred_body[0] / L;
            preferred_y_body = preferred_body[1] / L;
        }
        midline_stream << iteration_num << "," << loop_time << ","
                       << static_cast<int>(swimmer_data.stage) << "," << station << ","
                       << samples[index].xi << "," << samples[index].position(0) / L << ","
                       << samples[index].position(1) / L << "," << actual_body[0] / L << ","
                       << actual_body[1] / L << "," << target_x_body << ","
                       << target_y_body << "," << preferred_x_body << ","
                       << preferred_y_body << "," << samples[index].source << "\n";
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

        if (uses_active_strain(swimmer_data))
        {
            const double eta_abs_max =
                std::max(std::abs(swimmer_data.geometry.eta_min), std::abs(swimmer_data.geometry.eta_max));
            const double maximum_active_curvature =
                maximum_active_curvature_magnitude(swimmer_data);
            const bool uses_compatible_area_metric =
                swimmer_data.active_transverse_mode == ActiveTransverseMode::AREA_PRESERVING;
            const double metric_lower_argument =
                1.0 - (uses_compatible_area_metric ? 2.0 : 1.0) *
                          eta_abs_max * maximum_active_curvature;
            const double metric_upper_argument =
                1.0 + (uses_compatible_area_metric ? 2.0 : 1.0) *
                          eta_abs_max * maximum_active_curvature;
            const double raw_stretch_lower =
                uses_compatible_area_metric && metric_lower_argument > 0.0 ?
                    std::sqrt(metric_lower_argument) :
                    metric_lower_argument;
            const double raw_stretch_upper =
                uses_compatible_area_metric ?
                    std::sqrt(metric_upper_argument) :
                    metric_upper_argument;
            pout << "Maximum sampled active curvature magnitude = "
                 << maximum_active_curvature * swimmer_data.geometry.length << "/L.\n";
            pout << "Active curvature mode = "
                 << (swimmer_data.active_curvature_mode == ActiveCurvatureMode::GAIT_MATCHED ?
                         "GAIT_MATCHED" :
                         "STATIC_UNIFORM")
                 << ", transverse mode = "
                 << (swimmer_data.active_transverse_mode == ActiveTransverseMode::AREA_PRESERVING ?
                         "AREA_PRESERVING (compatible bending map)" :
                         "NONE")
                 << ".\n";
            pout << "Worst-case active axial stretch bound = [" << raw_stretch_lower << ","
                 << raw_stretch_upper << "].\n";
            if (metric_lower_argument <= 0.0)
            {
                TBOX_ERROR("The configured physical activation permits a nonpositive axial stretch. "
                           "Reduce ACTIVE_CURVATURE_SCALE or STATIC_ACTIVE_CURVATURE_TIMES_L.\n");
            }
            {
                constexpr unsigned int num_space_samples = 256;
                constexpr unsigned int num_time_samples = 256;
                const double end_time = swimmer_data.ramp_time + 1.0 / swimmer_data.frequency;
                double max_shear = 0.0;
                double max_Fa_norm = 0.0;
                double max_Fa_inv_norm = 0.0;
                for (unsigned int ti = 0; ti <= num_time_samples; ++ti)
                {
                    const double t = end_time * static_cast<double>(ti) / num_time_samples;
                    for (unsigned int si = 0; si <= num_space_samples; ++si)
                    {
                        const double s = swimmer_data.geometry.length *
                                         static_cast<double>(si) / num_space_samples;
                        const double kappa   = active_curvature_at_s(s, t, swimmer_data);
                        const double kappa_s = active_curvature_spatial_derivative(s, t, swimmer_data);
                        const double angle   = active_angle_at_s(s, t, swimmer_data);
                        for (const double eta_sign : { -1.0, 1.0 })
                        {
                            const double eta = eta_sign * eta_abs_max;
                            const double discriminant =
                                1.0 - (uses_compatible_area_metric ? 2.0 : 1.0) * eta * kappa;
                            if (!(discriminant > 0.0)) continue;
                            const ActiveMetricState m = active_metric_from_kinematics(
                                eta, kappa, kappa_s, angle, swimmer_data);
                            max_shear       = std::max(max_shear, std::abs(m.longitudinal_shear));
                            max_Fa_norm     = std::max(max_Fa_norm, frobenius_norm(m.gradient));
                            max_Fa_inv_norm = std::max(max_Fa_inv_norm, frobenius_norm(m.inverse));
                        }
                    }
                }
                pout << "Active metric startup bounds: max |gamma_a| = " << max_shear
                     << ", max ||Fa|| = " << max_Fa_norm
                     << ", max ||Fa_inv|| = " << max_Fa_inv_norm
                     << ", Frobenius cond(Fa) ~ " << max_Fa_norm * max_Fa_inv_norm / NDIM << ".\n";
            }
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
            if (!diagnostics_stream.is_open())
            {
                TBOX_ERROR("Unable to open DIAGNOSTICS_FILENAME: " << diagnostics_filename << "\n");
            }
            diagnostics_stream.precision(12);
            if (!from_restart)
            {
                diagnostics_stream
                    << "step,time,stage,J_total_min,J_total_max,J_elastic_min,J_elastic_max,"
                       "lambda_active_min,lambda_active_max,"
                       "P_matrix_mean,P_fiber_mean,P_shear_mean,"
                       "tracking_error_rms,tracking_error_max,tracking_error_rms_over_L,"
                       "tracking_error_max_over_L,tracking_velocity_error_rms,"
                       "tracking_velocity_error_max,tracking_velocity_error_rms_over_Lf,"
                       "tracking_velocity_error_max_over_Lf,"
                       "body_force_x,body_force_y,body_torque_z,body_power,"
                       "tail_root_lateral,tail_tip_lateral,tail_lateral_over_L,tail_pitch,"
                       "target_tail_root_lateral,target_tail_tip_lateral,target_tail_pitch,"
                       "actual_curvature_mid,actual_curvature_tail,target_curvature_mid,"
                       "target_curvature_tail,body_mid_lateral,target_body_mid_lateral,"
                       "actual_curvature_body,target_curvature_body,"
                       "actual_curvature_eta0_body,actual_curvature_eta0_mid,"
                       "actual_curvature_eta0_tail,J_active_min,J_active_max,"
                       "active_metric_longitudinal_shear_max\n";
            }
            if (swimmer_data.write_midline_data)
            {
                midline_stream.open(swimmer_data.midline_filename,
                                    from_restart ? ios_base::out | ios_base::app :
                                                   ios_base::out | ios_base::trunc);
                if (!midline_stream.is_open())
                {
                    TBOX_ERROR("Unable to open MIDLINE_FILENAME: "
                               << swimmer_data.midline_filename << "\n");
                }
                midline_stream.precision(12);
                if (!from_restart)
                {
                    midline_stream
                        << "step,time,stage,station_index,s_over_L,"
                           "x_lab_over_L,y_lab_over_L,x_body_over_L,y_body_over_L,"
                           "target_x_body_over_L,target_y_body_over_L,"
                           "preferred_x_body_over_L,preferred_y_body_over_L,"
                           "centerline_source\n";
                }
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
            diagnostics_stream.close();
            if (midline_stream.is_open()) midline_stream.close();
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
    static const std::array<double, 11> station_xi = {
        0.02, 0.35, 0.50, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98
    };
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
    double tracking_velocity_error_squared = 0.0;
    double tracking_velocity_error_max = 0.0;
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
    double lambda_active_min = std::numeric_limits<double>::max();
    double lambda_active_max = -std::numeric_limits<double>::max();
    double J_active_min = std::numeric_limits<double>::max();
    double J_active_max = -std::numeric_limits<double>::max();
    double active_metric_longitudinal_shear_max = 0.0;
    std::array<double, station_xi.size()> station_weight{};
    std::array<double, station_xi.size()> station_x{};
    std::array<double, station_xi.size()> station_y{};
    std::array<double, station_xi.size()> centerline_station_weight{};
    std::array<double, station_xi.size()> centerline_station_x{};
    std::array<double, station_xi.size()> centerline_station_y{};
    std::array<double, station_xi.size()> target_station_x{};
    std::array<double, station_xi.size()> target_station_y{};

    const int ml_num_output = swimmer_data.midline_num_stations;
    std::vector<double> ml_station_xi;
    std::size_t ml_frame_head_index = 0;
    std::size_t ml_frame_tangent_index = 0;
    std::vector<double> ml_section_weight;
    std::vector<double> ml_section_x;
    std::vector<double> ml_section_y;
    std::vector<double> ml_centerline_weight;
    std::vector<double> ml_centerline_x;
    std::vector<double> ml_centerline_y;
    if (swimmer_data.write_midline_data)
    {
        ml_station_xi.reserve(static_cast<std::size_t>(ml_num_output) + 2);
        for (int s = 0; s < ml_num_output; ++s)
            ml_station_xi.push_back(static_cast<double>(s) / (ml_num_output - 1));
        ml_frame_head_index = ml_station_xi.size();
        ml_station_xi.push_back(swimmer_data.midline_body_frame_head_xi);
        ml_frame_tangent_index = ml_station_xi.size();
        ml_station_xi.push_back(swimmer_data.midline_body_frame_tangent_xi);
        const std::size_t n = ml_station_xi.size();
        ml_section_weight.assign(n, 0.0);
        ml_section_x.assign(n, 0.0);
        ml_section_y.assign(n, 0.0);
        ml_centerline_weight.assign(n, 0.0);
        ml_centerline_x.assign(n, 0.0);
        ml_centerline_y.assign(n, 0.0);
    }

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
            const ActiveMetricState active_metric =
                active_metric_state(X, loop_time, swimmer_data);
            const MaterialResponse material =
                evaluate_material(FF, active_metric, swimmer_data, &X, loop_time);

            volume += weight;
            J_total_min = std::min(J_total_min, J_total);
            J_total_max = std::max(J_total_max, J_total);
            J_elastic_min = std::min(J_elastic_min, material.J_elastic);
            J_elastic_max = std::max(J_elastic_max, material.J_elastic);
            lambda_active_min = std::min(lambda_active_min, active_metric.axial_stretch);
            lambda_active_max = std::max(lambda_active_max, active_metric.axial_stretch);
            J_active_min = std::min(J_active_min, active_metric.J_active);
            J_active_max = std::max(J_active_max, active_metric.J_active);
            active_metric_longitudinal_shear_max =
                std::max(active_metric_longitudinal_shear_max,
                         std::abs(active_metric.longitudinal_shear));
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
                const VectorValue<double> velocity_error = target.velocity - velocity;
                const double velocity_error_squared = dot2(velocity_error, velocity_error);
                tracking_velocity_error_squared += velocity_error_squared * weight;
                tracking_velocity_error_max =
                    std::max(tracking_velocity_error_max, std::sqrt(velocity_error_squared));
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
                const double eta_abs = std::abs(reference_eta(X, swimmer_data));
                if (eta_abs < swimmer_data.diagnostic_centerline_half_thickness)
                {
                    const double eta_kernel =
                        1.0 - eta_abs / swimmer_data.diagnostic_centerline_half_thickness;
                    const double centerline_weight = eta_kernel * station_quadrature_weight;
                    centerline_station_weight[station] += centerline_weight;
                    centerline_station_x[station] += x(0) * centerline_weight;
                    centerline_station_y[station] += x(1) * centerline_weight;
                }
                if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
                {
                    target_station_x[station] += target.position(0) * station_quadrature_weight;
                    target_station_y[station] += target.position(1) * station_quadrature_weight;
                }
            }

            if (swimmer_data.write_midline_data)
            {
                const double ml_eta_abs = std::abs(reference_eta(X, swimmer_data));
                for (std::size_t ml_s = 0; ml_s < ml_station_xi.size(); ++ml_s)
                {
                    const double distance = std::abs(xi - ml_station_xi[ml_s]);
                    if (distance >= swimmer_data.midline_station_half_width) continue;
                    const double sk = (1.0 - distance / swimmer_data.midline_station_half_width) * weight;
                    ml_section_weight[ml_s] += sk;
                    ml_section_x[ml_s] += x(0) * sk;
                    ml_section_y[ml_s] += x(1) * sk;
                    if (ml_eta_abs < swimmer_data.midline_centerline_half_thickness)
                    {
                        const double ek =
                            (1.0 - ml_eta_abs / swimmer_data.midline_centerline_half_thickness) * sk;
                        ml_centerline_weight[ml_s] += ek;
                        ml_centerline_x[ml_s] += x(0) * ek;
                        ml_centerline_y[ml_s] += x(1) * ek;
                    }
                }
            }
        }
    }

    const std::array<double*, 10> sums = { &volume,
                                           &tracking_error_squared,
                                           &tracking_velocity_error_squared,
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
    IBTK_MPI::sumReduction(centerline_station_weight.data(), centerline_station_weight.size());
    IBTK_MPI::sumReduction(centerline_station_x.data(), centerline_station_x.size());
    IBTK_MPI::sumReduction(centerline_station_y.data(), centerline_station_y.size());
    IBTK_MPI::sumReduction(target_station_x.data(), target_station_x.size());
    IBTK_MPI::sumReduction(target_station_y.data(), target_station_y.size());
    IBTK_MPI::maxReduction(&tracking_error_max, 1);
    IBTK_MPI::maxReduction(&tracking_velocity_error_max, 1);
    IBTK_MPI::minReduction(&J_total_min, 1);
    IBTK_MPI::maxReduction(&J_total_max, 1);
    IBTK_MPI::minReduction(&J_elastic_min, 1);
    IBTK_MPI::maxReduction(&J_elastic_max, 1);
    IBTK_MPI::minReduction(&lambda_active_min, 1);
    IBTK_MPI::maxReduction(&lambda_active_max, 1);
    IBTK_MPI::minReduction(&J_active_min, 1);
    IBTK_MPI::maxReduction(&J_active_max, 1);
    IBTK_MPI::maxReduction(&active_metric_longitudinal_shear_max, 1);

    std::vector<MidlineSample> ml_samples;
    int ml_fallback_count = 0;
    if (swimmer_data.write_midline_data && !ml_station_xi.empty())
    {
        IBTK_MPI::sumReduction(ml_section_weight.data(), ml_section_weight.size());
        IBTK_MPI::sumReduction(ml_section_x.data(), ml_section_x.size());
        IBTK_MPI::sumReduction(ml_section_y.data(), ml_section_y.size());
        IBTK_MPI::sumReduction(ml_centerline_weight.data(), ml_centerline_weight.size());
        IBTK_MPI::sumReduction(ml_centerline_x.data(), ml_centerline_x.size());
        IBTK_MPI::sumReduction(ml_centerline_y.data(), ml_centerline_y.size());
        ml_fallback_count = assemble_midline_samples(
            ml_section_weight, ml_section_x, ml_section_y,
            ml_centerline_weight, ml_centerline_x, ml_centerline_y,
            ml_station_xi, iteration_num, loop_time, swimmer_data, ml_samples);
    }

    const double tracking_error_rms =
        volume > 0.0 ? std::sqrt(tracking_error_squared / volume) : 0.0;
    const double tracking_velocity_error_rms =
        volume > 0.0 ? std::sqrt(tracking_velocity_error_squared / volume) : 0.0;
    const double characteristic_velocity = swimmer_data.geometry.length * swimmer_data.frequency;
    const double P_matrix_mean = volume > 0.0 ? P_matrix_integral / volume : 0.0;
    const double P_fiber_mean = volume > 0.0 ? P_fiber_integral / volume : 0.0;
    const double P_shear_mean = volume > 0.0 ? P_shear_integral / volume : 0.0;

    std::array<libMesh::Point, station_xi.size()> stations;
    std::array<libMesh::Point, station_xi.size()> centerline_stations;
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
        if (centerline_station_weight[station] <= 0.0)
        {
            TBOX_ERROR("No diagnostic quadrature points found in the eta=0 strip near xi="
                       << station_xi[station]
                       << ". Increase DIAGNOSTIC_CENTERLINE_HALF_THICKNESS.\n");
        }
        centerline_stations[station](0) =
            centerline_station_x[station] / centerline_station_weight[station];
        centerline_stations[station](1) =
            centerline_station_y[station] / centerline_station_weight[station];
        if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
        {
            target_stations[station](0) = target_station_x[station] / station_weight[station];
            target_stations[station](1) = target_station_y[station] / station_weight[station];
        }
        else
        {
            target_stations[station] = active_preferred_centerline(
                station_xi[station] * swimmer_data.geometry.length, loop_time, swimmer_data);
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

    const double tail_root_lateral = lateral(stations[8]);
    const double tail_tip_lateral = lateral(stations[10]);
    const double tail_lateral_over_L =
        (tail_tip_lateral - lateral(stations[0])) / swimmer_data.geometry.length;
    const double tail_pitch = pitch(stations[8], stations[10]);
    const double body_mid_lateral = lateral(stations[2]);
    const double actual_curvature_body = curvature(stations[1], stations[2], stations[3]);
    const double actual_curvature_mid = curvature(stations[4], stations[5], stations[6]);
    const double actual_curvature_tail = curvature(stations[7], stations[8], stations[9]);
    const double actual_curvature_eta0_body =
        curvature(centerline_stations[1], centerline_stations[2], centerline_stations[3]);
    const double actual_curvature_eta0_mid =
        curvature(centerline_stations[4], centerline_stations[5], centerline_stations[6]);
    const double actual_curvature_eta0_tail =
        curvature(centerline_stations[7], centerline_stations[8], centerline_stations[9]);

    const double target_tail_root_lateral = lateral(target_stations[8]);
    const double target_tail_tip_lateral = lateral(target_stations[10]);
    const double target_tail_pitch = pitch(target_stations[8], target_stations[10]);
    const double target_body_mid_lateral = lateral(target_stations[2]);
    double target_curvature_body = 0.0;
    double target_curvature_mid = 0.0;
    double target_curvature_tail = 0.0;
    if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED)
    {
        target_curvature_body = curvature(target_stations[1], target_stations[2], target_stations[3]);
        target_curvature_mid = curvature(target_stations[4], target_stations[5], target_stations[6]);
        target_curvature_tail = curvature(target_stations[7], target_stations[8], target_stations[9]);
    }
    else
    {
        const libMesh::Point X_body =
            swimmer_data.geometry.head + station_xi[2] * swimmer_data.geometry.length *
                                             swimmer_data.geometry.tangent;
        const libMesh::Point X_mid =
            swimmer_data.geometry.head + station_xi[5] * swimmer_data.geometry.length *
                                             swimmer_data.geometry.tangent;
        const libMesh::Point X_tail =
            swimmer_data.geometry.head + station_xi[8] * swimmer_data.geometry.length *
                                             swimmer_data.geometry.tangent;
        const double s_body = reference_s(X_body, swimmer_data);
        const double s_mid = reference_s(X_mid, swimmer_data);
        const double s_tail = reference_s(X_tail, swimmer_data);
        target_curvature_body = active_curvature_at_s(s_body, loop_time, swimmer_data);
        target_curvature_mid = active_curvature_at_s(s_mid, loop_time, swimmer_data);
        target_curvature_tail = active_curvature_at_s(s_tail, loop_time, swimmer_data);
    }

    if (IBTK_MPI::getRank() == 0)
    {
        diagnostics_stream << iteration_num << "," << loop_time << ","
                           << static_cast<int>(swimmer_data.stage) << "," << J_total_min << ","
                           << J_total_max << "," << J_elastic_min << "," << J_elastic_max << ","
                           << lambda_active_min << "," << lambda_active_max << "," << P_matrix_mean << ","
                           << P_fiber_mean << "," << P_shear_mean << "," << tracking_error_rms << ","
                           << tracking_error_max << ","
                           << tracking_error_rms / swimmer_data.geometry.length << ","
                           << tracking_error_max / swimmer_data.geometry.length << ","
                           << tracking_velocity_error_rms << "," << tracking_velocity_error_max << ","
                           << tracking_velocity_error_rms / characteristic_velocity << ","
                           << tracking_velocity_error_max / characteristic_velocity << "," << body_force_x << ","
                           << body_force_y << "," << body_torque << "," << body_power << ","
                           << tail_root_lateral << "," << tail_tip_lateral << "," << tail_lateral_over_L << ","
                           << tail_pitch << "," << target_tail_root_lateral << ","
                           << target_tail_tip_lateral << "," << target_tail_pitch << ","
                           << actual_curvature_mid << "," << actual_curvature_tail << ","
                           << target_curvature_mid << "," << target_curvature_tail << ","
                           << body_mid_lateral << "," << target_body_mid_lateral << ","
                           << actual_curvature_body << "," << target_curvature_body << ","
                           << actual_curvature_eta0_body << "," << actual_curvature_eta0_mid << ","
                           << actual_curvature_eta0_tail << "," << J_active_min << ","
                           << J_active_max << "," << active_metric_longitudinal_shear_max << "\n";
        diagnostics_stream.flush();
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
    if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED &&
        tracking_error_max / swimmer_data.geometry.length > swimmer_data.max_tracking_error_over_L)
    {
        TBOX_ERROR("Penalty tracking error limit exceeded at step "
                   << iteration_num << ", time = " << loop_time << ": max error/L = "
                   << tracking_error_max / swimmer_data.geometry.length
                   << ", allowed maximum = " << swimmer_data.max_tracking_error_over_L << ".\n");
    }
    if (swimmer_data.stage == SwimmerStage::PENALTY_PRESCRIBED &&
        tracking_velocity_error_max / characteristic_velocity >
            swimmer_data.max_tracking_velocity_error_over_Lf)
    {
        TBOX_ERROR("Penalty velocity tracking limit exceeded at step "
                   << iteration_num << ", time = " << loop_time << ": max |U_target-U|/(L*f) = "
                   << tracking_velocity_error_max / characteristic_velocity
                   << ", allowed maximum = " << swimmer_data.max_tracking_velocity_error_over_Lf << ".\n");
    }
}

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

// Config files
#include <SAMRAI_config.h>

// Headers for basic PETSc functions
#include <petscsys.h>

#include <boost/multi_array.hpp>

#include <fstream>
#include <iomanip>

// Headers for basic SAMRAI objects
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

// Headers for basic libMesh objects
#include <libmesh/boundary_info.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>

// Headers for application-specific algorithm/data structure objects
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFECentroidPostProcessor.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

// Set up application namespace declarations
#include <ibamr/app_namespaces.h>

// ---------------------------------------------------------------------
// Third-stage kinematic audit:
// target-only, fixed y/theta, free global x translation, no
// self-propulsion, no active stress. The target still prescribes shape,
// but the global x frame is refreshed from the current coupled
// configuration each step.
// ---------------------------------------------------------------------

namespace ModelData
{

static double kappa_s = 1.0e6;

// Elastic stress parameters (keep as simple passive support)
static double c1_s = 0.05;
static double p0_s = 0.0;
static double beta_s = 0.0;

// -------------------------------
// Target-only audit parameters
// -------------------------------
static double xcom_fixed = 0.25;
static double xcom_target_current = xcom_fixed;
static double ycom_fixed = 0.00;
static double theta_fixed = 0.0;    // 固定 target frame 的 y/theta，x 由当前构型决定。

static double fish_length = 1.0;
static double wave_amplitude = 0.05;      // loaded from input2d
static double wave_frequency = 1.0;       // Hz, loaded from input2d
static double wave_omega = 2.0 * M_PI * wave_frequency;
static double wave_lambda = 1.0;          // loaded from input2d
static double wave_k = 2.0 * M_PI / wave_lambda;
static double wave_phase0 = 0.0;          // loaded from input2d

// Leading point of the reference body centerline
static double x_leading = -0.1;
static double y_center0 = 0.0;

static std::string target_wave_audit_file = "target_wave_audit.curve";
static std::ofstream target_wave_audit_stream;

// -------------------------------
// Simple amplitude envelope:
// head small, tail large
// A(s) = A0 * (s/L)
// -------------------------------
inline double
amplitude_envelope(double s, double L, double A0)
{
    if (s <= 0.0) return 0.0;
    if (s >= L) return A0;
    const double xi = s / L;
    return A0 * xi;
}

inline double
amplitude_envelope_derivative(double s, double L, double A0)
{
    if (s <= 0.0 || s >= L) return 0.0;
    return A0 / L;
}

// -------------------------------
// Traveling-wave body kinematics
// y(s,t) = A(s) * sin(omega*t - k*s + phase0)
// -------------------------------
inline double
body_wave_y(double s,
            double t,
            double L,
            double A0,
            double omega,
            double k_wave,
            double phase0)
{
    const double A = amplitude_envelope(s, L, A0);
    return A * std::sin(omega * t - k_wave * s + phase0);
}

// -------------------------------
// Centerline slope
// dy/ds = A'(s) * sin(omega*t - k*s + phase0)
//       - k * A(s) * cos(omega*t - k*s + phase0)
// -------------------------------
inline double
body_wave_dyds(double s,
               double t,
               double L,
               double A0,
               double omega,
               double k_wave,
               double phase0)
{
    const double A = amplitude_envelope(s, L, A0);
    const double dA_ds = amplitude_envelope_derivative(s, L, A0);
    const double phase = omega * t - k_wave * s + phase0;
    return dA_ds * std::sin(phase) - k_wave * A * std::cos(phase);
}

// -------------------------------
// Time derivative of body wave
// dy/dt = omega * A(s) * cos(omega*t - k*s + phase0)
// -------------------------------
inline double
body_wave_v(double s,
            double t,
            double L,
            double A0,
            double omega,
            double k_wave,
            double phase0)
{
    const double A = amplitude_envelope(s, L, A0);
    return omega * A * std::cos(omega * t - k_wave * s + phase0);
}

inline double
target_wave_y_at_s(double s_body, double time)
{
    return body_wave_y(s_body,
                       time,
                       fish_length,
                       wave_amplitude,
                       wave_omega,
                       wave_k,
                       wave_phase0);
}

inline void
initialize_target_wave_audit_stream()
{
    if (target_wave_audit_stream.is_open()) target_wave_audit_stream.close();

    target_wave_audit_stream.open(target_wave_audit_file.c_str(), std::ios::out | std::ios::trunc);
    if (!target_wave_audit_stream.is_open())
    {
        TBOX_ERROR("Could not open target wave audit file: " << target_wave_audit_file << "\n");
    }

    target_wave_audit_stream << std::scientific << std::setprecision(12);
    target_wave_audit_stream << "# time tail_s tail_y s_0.1 y_0.1 s_0.3 y_0.3 s_0.5 y_0.5 s_0.7 y_0.7 s_0.9 y_0.9\n";
    target_wave_audit_stream.flush();
}

inline void
audit_target_wave(double time)
{
    const double s_tail = fish_length;
    const double s_01 = 0.1 * fish_length;
    const double s_03 = 0.3 * fish_length;
    const double s_05 = 0.5 * fish_length;
    const double s_07 = 0.7 * fish_length;
    const double s_09 = 0.9 * fish_length;

    if (!target_wave_audit_stream.is_open()) initialize_target_wave_audit_stream();

    target_wave_audit_stream << time
                             << ' ' << s_tail << ' ' << target_wave_y_at_s(s_tail, time)
                             << ' ' << s_01 << ' ' << target_wave_y_at_s(s_01, time)
                             << ' ' << s_03 << ' ' << target_wave_y_at_s(s_03, time)
                             << ' ' << s_05 << ' ' << target_wave_y_at_s(s_05, time)
                             << ' ' << s_07 << ' ' << target_wave_y_at_s(s_07, time)
                             << ' ' << s_09 << ' ' << target_wave_y_at_s(s_09, time)
                             << '\n';
    target_wave_audit_stream.flush();
}

// -------------------------------
// Target generator with prescribed y/theta and supplied frame x-position
// Reference configuration -> prescribed body frame ->
// centerline wave + local normal offset -> map back to lab frame
// -------------------------------
inline void
compute_eel_target_fixed_pose(const libMesh::Point& X,
                              double time,
                              double xcom_ref,
                              double ycom_ref,
                              double theta_ref,
                              double xcom_tar,
                              double ycom_tar,
                              double theta_tar,
                              double xlead,
                              double ycenter0,
                              double L,
                              double A0,
                              double omega,
                              double k_wave,
                              double phase0,
                              double& xtar,
                              double& ytar)
{
    // 1) reference point -> prescribed body frame
    const double dx_ref = X(0) - xcom_ref;
    const double dy_ref = X(1) - ycom_ref;

    const double c_ref = std::cos(theta_ref);
    const double s_ref = std::sin(theta_ref);

    const double xhat_ref =  c_ref * dx_ref + s_ref * dy_ref;
    const double yhat_ref = -s_ref * dx_ref + c_ref * dy_ref;

    // 2) leading point -> prescribed body frame
    const double dx_lead = xlead - xcom_ref;
    const double dy_lead = ycenter0 - ycom_ref;
    const double xhat_lead = c_ref * dx_lead + s_ref * dy_lead;
    const double yhat_lead = -s_ref * dx_lead + c_ref * dy_lead;

    // 3) body-frame coordinates relative to the undeformed centerline
    const double s_body = xhat_ref - xhat_lead;
    const double eta_body = yhat_ref - yhat_lead;

    // 4) deformed centerline and local Frenet frame in the body frame
    const double ywave = body_wave_y(s_body, time, L, A0, omega, k_wave, phase0);
    const double dywave_ds = body_wave_dyds(s_body, time, L, A0, omega, k_wave, phase0);

    const double tangent_scale = std::sqrt(1.0 + dywave_ds * dywave_ds);
    const double tx = 1.0 / tangent_scale;
    const double ty = dywave_ds / tangent_scale;
    const double nx = -ty;
    const double ny = tx;

    const double xhat_center = xhat_lead + s_body;
    const double yhat_center = yhat_lead + ywave;

    const double xhat_tar = xhat_center + eta_body * nx;
    const double yhat_tar = yhat_center + eta_body * ny;

    // 5) map back to lab frame with released global x but fixed y/theta
    const double c_tar = std::cos(theta_tar);
    const double s_tar = std::sin(theta_tar);
    xtar = xcom_tar + c_tar * xhat_tar - s_tar * yhat_tar;
    ytar = ycom_tar + s_tar * xhat_tar + c_tar * yhat_tar;
}

// -------------------------------
// Target-only tether force:
// target prescribes shape and fixed y/theta, but not global x translation.
// -------------------------------
void
target_force_function(libMesh::VectorValue<double>& F,
                      const libMesh::TensorValue<double>& /*FF*/,
                      const libMesh::Point& x,
                      const libMesh::Point& X,
                      libMesh::Elem* const /*elem*/,
                      const std::vector<const std::vector<double>*>& /*system_var_data*/,
                      const std::vector<const std::vector<libMesh::VectorValue<double> >*>& /*system_grad_var_data*/,
                      double time,
                      void* /*ctx*/)
{
    double xtar, ytar;
    compute_eel_target_fixed_pose(X,
                                  time,
                                  xcom_fixed,
                                  ycom_fixed,
                                  theta_fixed,
                                  xcom_target_current,
                                  ycom_fixed,
                                  theta_fixed,
                                  x_leading,
                                  y_center0,
                                  fish_length,
                                  wave_amplitude,
                                  wave_omega,
                                  wave_k,
                                  wave_phase0,
                                  xtar,
                                  ytar);

    libMesh::Point X_target;
    X_target(0) = xtar;
    X_target(1) = ytar;

    F = kappa_s * (X_target - x);
}

// -------------------------------
// Passive elasticity (keep simple)
// -------------------------------
void
PK1_dev_stress_function(TensorValue<double>& PP,
                        const TensorValue<double>& FF,
                        const libMesh::Point& /*X*/,
                        const libMesh::Point& /*s*/,
                        Elem* const /*elem*/,
                        const std::vector<const std::vector<double>*>& /*var_data*/,
                        const std::vector<const std::vector<VectorValue<double> >*>& /*grad_var_data*/,
                        double /*time*/,
                        void* /*ctx*/)
{
    PP = 2.0 * c1_s * FF;
}

void
PK1_dil_stress_function(TensorValue<double>& PP,
                        const TensorValue<double>& FF,
                        const libMesh::Point& /*X*/,
                        const libMesh::Point& /*s*/,
                        Elem* const /*elem*/,
                        const std::vector<const std::vector<double>*>& /*var_data*/,
                        const std::vector<const std::vector<VectorValue<double> >*>& /*grad_var_data*/,
                        double /*time*/,
                        void* /*ctx*/)
{
    PP = 2.0 * (-p0_s + beta_s * std::log(FF.det())) * tensor_inverse_transpose(FF, NDIM);
}

} // namespace ModelData
using namespace ModelData;

// Function prototypes
double compute_current_xcom(EquationSystems* equation_systems,
                            const std::string& coords_system_name,
                            MeshBase& mesh);
void update_target_xcom_cache(EquationSystems* equation_systems,
                              const std::string& coords_system_name,
                              MeshBase& mesh);
void output_data(Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
                 Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
                 Mesh& mesh,
                 EquationSystems* equation_systems,
                 const int iteration_num,
                 const double loop_time,
                 const string& data_dump_dirname);

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/

int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    const LibMeshInit& init = ibtk_init.getLibMeshInit();

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();
#ifdef LIBMESH_HAVE_EXODUS_API
        const bool uses_exodus = dump_viz_data && !app_initializer->getExodusIIFilename().empty();
#else
        const bool uses_exodus = false;
        if (!app_initializer->getExodusIIFilename().empty())
        {
            plog << "WARNING: libMesh was compiled without Exodus support, so no "
                 << "Exodus output will be written in this program.\n";
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
        if (dump_postproc_data && (postproc_data_dump_interval > 0) && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create a simple FE mesh.
        //
        // Note that boundary condition data must be registered with each FE
        // system before calling IBFEMethod::initializeFEData().
        Mesh mesh(init.comm(), NDIM);
        //Mesh mesh(NDIM);
            const double dx = input_db->getDouble("DX");
            const double ds = input_db->getDouble("MFAC")*dx;
        string elem_type = input_db->getString("ELEM_TYPE");
        mesh.read("fish2d.msh");

        mesh.prepare_for_use();
        
        pout << "mesh_dimension=" << mesh.mesh_dimension()
        << ", spatial_dimension=" << mesh.spatial_dimension() << "\n";

        c1_s = input_db->getDouble("C1_S");
        p0_s = input_db->getDouble("P0_S");
        beta_s = input_db->getDouble("BETA_S");
        kappa_s = input_db->getDouble("Kappa_S");

        xcom_fixed = input_db->getDouble("Xcom_fixed");
        ycom_fixed  = input_db->getDouble("Ycom_fixed");
        theta_fixed = input_db->getDouble("Theta_fixed");

        fish_length = input_db->getDouble("FISH_LENGTH");
        wave_amplitude = input_db->getDouble("WAVE_AMPLITUDE");
        wave_frequency = input_db->getDouble("WAVE_FREQUENCY");
        wave_lambda = input_db->getDouble("WAVE_LAMBDA");
        wave_phase0 = input_db->getDouble("WAVE_PHASE0");
        x_leading = input_db->getDouble("X_LEADING");
        y_center0 = input_db->getDouble("Y_center0");
        target_wave_audit_file = input_db->getStringWithDefault("TARGET_WAVE_AUDIT_FILE", "target_wave_audit.curve");

        if (fish_length <= 0.0) TBOX_ERROR("FISH_LENGTH must be positive.\n");
        if (wave_lambda <= 0.0) TBOX_ERROR("WAVE_LAMBDA must be positive.\n");

        wave_omega = 2.0 * M_PI * wave_frequency;
        wave_k = 2.0 * M_PI / wave_lambda;

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
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
            TBOX_ERROR("Unsupported solver type: " << solver_type << "\n"
                                                   << "Valid options are: COLLOCATED, STAGGERED");
        }
        Pointer<IBFEMethod> ib_method_ops =
            new IBFEMethod("IBFEMethod",
                           app_initializer->getComponentDatabase("IBFEMethod"),
                           &mesh,
                           app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"),
                           /*register_for_restart*/ true,
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

        // Configure the IBFE solver.
        IBFEMethod::PK1StressFcnData PK1_dev_stress_data(PK1_dev_stress_function);
        IBFEMethod::PK1StressFcnData PK1_dil_stress_data(PK1_dil_stress_function);
        IBFEMethod::LagBodyForceFcnData target_force_data(target_force_function);   // target force function

        PK1_dev_stress_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault("PK1_DEV_QUAD_ORDER", "THIRD"));
        PK1_dil_stress_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault("PK1_DIL_QUAD_ORDER", "FIRST"));
        ib_method_ops->registerPK1StressFunction(PK1_dev_stress_data);
        ib_method_ops->registerPK1StressFunction(PK1_dil_stress_data);
        ib_method_ops->registerLagBodyForceFunction(target_force_data);         // Configure target forces.
        
        ib_method_ops->initializeFEEquationSystems();

        FEDataManager* fe_data_manager = ib_method_ops->getFEDataManager();
        EquationSystems* equation_systems = fe_data_manager->getEquationSystems();
        const std::string coords_system_name = ib_method_ops->getCurrentCoordinatesSystemName();

        // Set up post processor to recover computed stresses.
        Pointer<IBFEPostProcessor> ib_post_processor =
            new IBFECentroidPostProcessor("IBFEPostProcessor", fe_data_manager);

        ib_post_processor->registerTensorVariable("FF", MONOMIAL, CONSTANT, IBFEPostProcessor::FF_fcn);

        std::pair<IBTK::TensorMeshFcnPtr, void*> PK1_dev_stress_fcn_data(PK1_dev_stress_function, nullptr);
        ib_post_processor->registerTensorVariable("sigma_dev",
                                                  MONOMIAL,
                                                  CONSTANT,
                                                  IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                                                  std::vector<SystemData>(),
                                                  &PK1_dev_stress_fcn_data);

        std::pair<IBTK::TensorMeshFcnPtr, void*> PK1_dil_stress_fcn_data(PK1_dil_stress_function, nullptr);
        ib_post_processor->registerTensorVariable("sigma_dil",
                                                  MONOMIAL,
                                                  CONSTANT,
                                                  IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                                                  std::vector<SystemData>(),
                                                  &PK1_dil_stress_fcn_data);

        Pointer<hier::Variable<NDIM> > p_var = navier_stokes_integrator->getPressureVariable();
        Pointer<VariableContext> p_current_ctx = navier_stokes_integrator->getCurrentContext();
        HierarchyGhostCellInterpolation::InterpolationTransactionComponent p_ghostfill(
            /*data_idx*/ -1, "LINEAR_REFINE", /*use_cf_bdry_interpolation*/ false, "CONSERVATIVE_COARSEN", "LINEAR");
        FEDataManager::InterpSpec p_interp_spec("PIECEWISE_LINEAR",
                                                QGAUSS,
                                                FIFTH,
                                                /*use_adaptive_quadrature*/ false,
                                                /*point_density*/ 2.0,
                                                /*use_consistent_mass_matrix*/ true,
                                                /*use_nodal_quadrature*/ false,
                                                /*allow_rules_with_negative_weights*/ false);
        ib_post_processor->registerInterpolatedScalarEulerianVariable(
            "p_f", LAGRANGE, FIRST, p_var, p_current_ctx, p_ghostfill, p_interp_spec);

        // Create Eulerian initial condition specification objects.
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

        // Create Eulerian boundary condition specification objects (when necessary).
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const std::string bc_coefs_name = "u_bc_coefs_" + std::to_string(d);

            const std::string bc_coefs_db_name = "VelocityBcCoefs_" + std::to_string(d);

            u_bc_coefs[d] = new muParserRobinBcCoefs(
                bc_coefs_name, app_initializer->getComponentDatabase(bc_coefs_db_name), grid_geometry);
        }
        navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);

        // Create Eulerian body force function specification objects.
        if (input_db->keyExists("ForcingFunction"))
        {
            Pointer<CartGridFunction> f_fcn = new muParserCartGridFunction(
                "f_fcn", app_initializer->getComponentDatabase("ForcingFunction"), grid_geometry);
            time_integrator->registerBodyForceFunction(f_fcn);
        }

        // Set up visualization plot file writers.
        Pointer<VisItDataWriter<NDIM> > visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            time_integrator->registerVisItDataWriter(visit_data_writer);
        }
        std::unique_ptr<ExodusII_IO> exodus_io = uses_exodus ? std::make_unique<ExodusII_IO>(mesh) : nullptr;

        // Check to see if this is a restarted run to append current exodus files
        if (uses_exodus)
        {
            const bool from_restart = RestartManager::getManager()->isFromRestart();
            exodus_io->append(from_restart);
        }

        // Initialize hierarchy configuration and data on all patches.
        ib_method_ops->initializeFEData();
        if (ib_post_processor) ib_post_processor->initializeFEData();
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);
        update_target_xcom_cache(equation_systems, coords_system_name, mesh);

        // Deallocate initialization objects.
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Write out initial visualization data.
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        if (dump_viz_data)
        {
            pout << "\n\nWriting visualization files...\n\n";
            if (uses_visit)
            {
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            }
            if (uses_exodus)
            {
                if (ib_post_processor) ib_post_processor->postProcessData(loop_time);
                exodus_io->write_timestep(
                    exodus_filename, *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
            }
        }

        initialize_target_wave_audit_stream();

        // Main time step loop.
        double loop_time_end = time_integrator->getEndTime();
        double dt = 0.0;
        const int audit_interval = input_db->getIntegerWithDefault("AUDIT_INTERVAL", 20);
        while (!IBTK::rel_equal_eps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();

            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            if (audit_interval > 0 && iteration_num % audit_interval == 0)
            {
                audit_target_wave(loop_time);
            }

            update_target_xcom_cache(equation_systems, coords_system_name, mesh);
            dt = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt);
            loop_time += dt;
            update_target_xcom_cache(equation_systems, coords_system_name, mesh);

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            // At specified intervals, write visualization and restart files,
            // print out timer data, and store hierarchy data for post
            // processing.
            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                if (uses_visit)
                {
                    time_integrator->setupPlotData();
                    visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                }
                if (uses_exodus)
                {
                    if (ib_post_processor) ib_post_processor->postProcessData(loop_time);
                    exodus_io->write_timestep(
                        exodus_filename, *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
                }
            }
            if (dump_restart_data && (iteration_num % restart_dump_interval == 0 || last_step))
            {
                pout << "\nWriting restart files...\n\n";
                RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
                ib_method_ops->writeFEDataToRestartFile(restart_dump_dirname, iteration_num);
            }
            if (dump_timer_data && (iteration_num % timer_dump_interval == 0 || last_step))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
            if (dump_postproc_data && (iteration_num % postproc_data_dump_interval == 0 || last_step))
            {
                pout << "\nWriting state data...\n\n";
                output_data(patch_hierarchy,
                            navier_stokes_integrator,
                            mesh,
                            equation_systems,
                            iteration_num,
                            loop_time,
                            postproc_data_dump_dirname);
            }
        }

        if (target_wave_audit_stream.is_open()) target_wave_audit_stream.close();

        // Cleanup Eulerian boundary condition specification objects (when
        // necessary).
        for (unsigned int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];

    } // cleanup dynamically allocated objects prior to shutdown
} // main

double
compute_current_xcom(EquationSystems* equation_systems, const std::string& coords_system_name, MeshBase& mesh)
{
    const unsigned int dim = mesh.mesh_dimension();
    System& X_system = equation_systems->get_system<System>(coords_system_name);
    NumericVector<double>* X_vec = X_system.solution.get();
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& dof_map = X_system.get_dof_map();

    std::vector<std::vector<unsigned int> > dof_indices(NDIM);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, dim, THIRD);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<double> >& phi = fe->get_phi();

    boost::multi_array<double, 2> X_node;
    libMesh::Point x;

    double area = 0.0;
    double mx = 0.0;
    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d) dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        const unsigned int n_qp = qrule->n_points();
        for (unsigned int qp = 0; qp < n_qp; ++qp)
        {
            interpolate(x, qp, X_node, phi);
            area += JxW[qp];
            mx += x(0) * JxW[qp];
        }
    }

    double local_first[2] = { area, mx };
    IBTK_MPI::sumReduction(local_first, 2);
    const double area_tot = local_first[0] > 0.0 ? local_first[0] : 1.0;
    return local_first[1] / area_tot;
}

void
update_target_xcom_cache(EquationSystems* equation_systems, const std::string& coords_system_name, MeshBase& mesh)
{
    xcom_target_current = compute_current_xcom(equation_systems, coords_system_name, mesh);
}

void
output_data(Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
            Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
            Mesh& mesh,
            EquationSystems* equation_systems,
            const int iteration_num,
            const double loop_time,
            const string& data_dump_dirname)
{
    plog << "writing hierarchy data at iteration " << iteration_num << " to disk" << endl;
    plog << "simulation time is " << loop_time << endl;

    // Write Cartesian data.
    string file_name = data_dump_dirname + "/" + "hier_data.";
    char temp_buf[128];
    std::snprintf(temp_buf, sizeof(temp_buf), "%05d.samrai.%05d", iteration_num, IBTK_MPI::getRank());
    file_name += temp_buf;
    Pointer<HDFDatabase> hier_db = new HDFDatabase("hier_db");
    hier_db->create(file_name);
    VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
    ComponentSelector hier_data;
    hier_data.setFlag(var_db->mapVariableAndContextToIndex(navier_stokes_integrator->getVelocityVariable(),
                                                           navier_stokes_integrator->getCurrentContext()));
    hier_data.setFlag(var_db->mapVariableAndContextToIndex(navier_stokes_integrator->getPressureVariable(),
                                                           navier_stokes_integrator->getCurrentContext()));
    patch_hierarchy->putToDatabase(hier_db->putDatabase("PatchHierarchy"), hier_data);
    hier_db->putDouble("loop_time", loop_time);
    hier_db->putInteger("iteration_num", iteration_num);
    hier_db->close();

    // Write Lagrangian data.
    file_name = data_dump_dirname + "/" + "fe_mesh.";
    std::snprintf(temp_buf, sizeof(temp_buf), "%05d", iteration_num);
    file_name += temp_buf;
    file_name += ".xda";
    mesh.write(file_name);
    file_name = data_dump_dirname + "/" + "fe_equation_systems.";
    std::snprintf(temp_buf, sizeof(temp_buf), "%05d", iteration_num);
    file_name += temp_buf;
    equation_systems->write(file_name, (EquationSystems::WRITE_DATA | EquationSystems::WRITE_ADDITIONAL_DATA));
    return;
} // output_data

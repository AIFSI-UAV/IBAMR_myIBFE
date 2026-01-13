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
// add new .e file
// Config files
#include <SAMRAI_config.h>

// Headers for basic PETSc functions
#include <petscsys.h>

// Headers for basic SAMRAI objects
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

// Headers for basic libMesh objects
#include <libmesh/boundary_info.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_triangle_interface.h>
#include <libmesh/replicated_mesh.h>

// Headers for application-specific algorithm/data structure objects
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFECentroidPostProcessor.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/BoxPartitioner.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <boost/multi_array.hpp>

// Set up application namespace declarations
#include <ibamr/app_namespaces.h>
//add new file
#include <array>
#include <algorithm>
#include <cmath>
#include <libmesh/mesh_communication.h>

// Elasticity model data.  ModelData 命名空间：几何映射（初始位置）+ 两个 PK1 stress（dev/dil）
namespace ModelData
{
// Coordinate mapping function.
// ------------------------------
// 1.1 初始坐标映射：X = X(s)
//     这里做的事情非常简单：把结构整体平移到 (0.6,0.5) 附近
//     注意：s 是 reference coordinates（参考构型坐标）
//           X 是 current/physical coordinates（当前/物理坐标）
// ------------------------------
// 默认值保持与你现在一致：2D (0.6,0.5), 3D (0.6,0.5,0.5)
static std::array<double, NDIM> X_shift = []{
    std::array<double, NDIM> a{};
    a[0] = 0.6;
    a[1] = 0.5;
#if (NDIM == 3)
    a[2] = 0.5;
#endif
    return a;
}();

void 
coordinate_mapping_function(libMesh::Point& X, const libMesh::Point& s, void* /*ctx*/)
{
    X = s;
    for (unsigned int d = 0; d < NDIM; ++d) X(d) += X_shift[d];
    return;
}
// coordinate_mapping_function
// ------------------------------
// 1.2 材料参数（示例用 static 全局变量 + 从 input 读入）
//     c1_s   ：剪切相关系数（dev 部分）
//     p0_s   ：类似“参考压力/拉格朗日乘子常数项”（dil 部分）
//     beta_s ：体积相关惩罚强度（dil 部分）
// ------------------------------
// Stress tensor functions.
static double c1_s = 0.05;
static double p0_s = 0.0;
static double beta_s = 0.0;

// ------------------------------
// 1.3 PK1 dev（偏）应力：P_dev = 2*c1*F
//     FF = F = ∂X/∂s（变形梯度）
//     PP = P = 一阶 Piola–Kirchhoff 应力
//
//     这对应于一种非常简化的 neo-Hookean 形式的 dev 部分。
//     未来你加入 active stress 时，通常会：
//        P_total = P_dev + P_dil + P_active
// ------------------------------
void
PK1_dev_stress_function(TensorValue<double>& PP,
                        const TensorValue<double>& FF,
                        const libMesh::Point& /*X*/,
                        const libMesh::Point& /*s*/,
                        Elem* const /*elem*/,
                        const vector<const vector<double>*>& /*var_data*/,
                        const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                        double /*time*/,
                        void* /*ctx*/)
{
    PP = 2.0 * c1_s * FF;
    return;
} // PK1_dev_stress_function

// ------------------------------
// 1.4 PK1 dil（体积）应力：P_dil = 2*(-p0 + beta*log(det(F))) * F^{-T}
//     - det(F) = J（体积比）
//     - log(J) 常用于 quasi-incompressible 的能量项
//     - tensor_inverse_transpose(FF, NDIM) 即 F^{-T}
//
//     这部分控制“可压缩/不可压缩”倾向：
//       beta_s 越大，越接近体积不变（J≈1）
// ------------------------------
void
PK1_dil_stress_function(TensorValue<double>& PP,
                        const TensorValue<double>& FF,
                        const libMesh::Point& /*X*/,
                        const libMesh::Point& /*s*/,
                        Elem* const /*elem*/,
                        const vector<const vector<double>*>& /*var_data*/,
                        const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                        double time,
                        void* ctx)
{
    PP = 2.0 * (-p0_s + beta_s * log(FF.det())) * tensor_inverse_transpose(FF, NDIM);
    return;
} // PK1_dil_stress_function

// ------------------------------
// Active stress: 外环 + 环向纤维 + 周期激活
// ------------------------------
struct ActiveStressCtx
{
    bool   enable      = false;

    // 时间激活
    double T_max       = 0.0;   // 主动张力幅值
    double period      = 1.0;   // 周期
    double t_on        = 0.0;   // 周期内激活起点
    double t_off       = 0.5;   // 周期内激活终点（t_off > t_on）
    double phase       = 0.0;   // 相位偏移（可选）
    double ramp_time   = 0.0;   // 前 ramp_time 内线性爬升到满幅（可选，避免瞬时冲击）

    // 空间激活：外环带 [r_in, r_out]
    double r_in        = 0.16;  // 外环内半径（参考构型）
    double r_out       = 0.20;  // 外环外半径（参考构型）
    double eps_r       = 0.005; // 平滑宽度（tanh 平滑）
};

static ActiveStressCtx active_ctx;

static inline double smooth_step_tanh(double x, double eps)
{
    if (eps <= 0.0) return (x >= 0.0) ? 1.0 : 0.0;
    return 0.5 * (1.0 + std::tanh(x / eps));
}

static inline double band_mask(double r, double r_in, double r_out, double eps)
{
    // 1 for r in [r_in, r_out], 0 outside, with smooth transitions
    const double m1 = smooth_step_tanh(r - r_in, eps);
    const double m2 = smooth_step_tanh(r_out - r, eps);
    return m1 * m2;
}

static inline double activation_time(double time, const ActiveStressCtx& a)
{
    if (!a.enable || a.T_max == 0.0) return 0.0;
    if (a.period <= 0.0) return 0.0;

    // 相位
    double t = time + a.phase;

    // 周期内相位
    double tau = std::fmod(t, a.period);
    if (tau < 0.0) tau += a.period;

    if (!(a.t_off > a.t_on)) return 0.0;
    if (tau < a.t_on || tau > a.t_off) return 0.0;

    // 平滑激活：sin^2(pi*xi) in [0,1]
    const double xi = (tau - a.t_on) / (a.t_off - a.t_on);
    double alpha = std::sin(M_PI * xi);
    alpha = alpha * alpha;

    // 可选 ramp（前 ramp_time 内从 0 -> 1）
    if (a.ramp_time > 0.0)
    {
        const double rfac = std::min(1.0, std::max(0.0, time / a.ramp_time));
        alpha *= rfac;
    }
    return alpha;
}

// 额外注册的第三个 PK1：只返回“主动项”P_active
void PK1_active_stress_function(TensorValue<double>& PP,
                                const TensorValue<double>& FF,
                                const libMesh::Point& /*X*/,
                                const libMesh::Point& s,
                                Elem* const /*elem*/,
                                const vector<const vector<double>*>& /*var_data*/,
                                const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                                double /*time*/,
                                void* /*ctx*/)
{
    const ActiveStressCtx& a = *static_cast<ActiveStressCtx*>(ctx);

    // 默认 0（注意：该函数只贡献 active 部分）
    PP = 0.0;

    if (!a.enable) return;

    // reference radius
    const double x = s(0);
    const double y = s(1);
    const double r = std::sqrt(x*x + y*y);
    if (r < 1.0e-12) return;

    // 外环带 mask
    const double m_space = band_mask(r, a.r_in, a.r_out, a.eps_r);
    if (m_space <= 0.0) return;

    // 周期激活
    const double alpha = activation_time(time, a);
    if (alpha <= 0.0) return;

    const double Tact = a.T_max * alpha * m_space;

    // 环向纤维（reference）：a0 = (-y, x)/r
    libMesh::VectorValue<double> a0(-y / r, x / r);
    // current fiber direction ~ F * a0
    libMesh::VectorValue<double> Fa = FF * a0;

    // P_active = T * (F a0) ⊗ a0
    for (unsigned int i = 0; i < NDIM; ++i)
        for (unsigned int j = 0; j < NDIM; ++j)
            PP(i,j) = Tact * Fa(i) * a0(j);
}

} // namespace ModelData

using namespace ModelData;
// Function prototypes
// ============================================================================
// 2) 输出函数原型：用于“后处理数据 dump”（HDF + xda + equation systems）
// ============================================================================
void output_data(Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
                 Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
                 MeshBase& mesh,
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

//main前半段：读取 input、生成 FE mesh（拉格朗日网格）
int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    // =========================================================================
    // 3) 初始化：IBTKInit 会初始化 PETSc / SAMRAI / libMesh，并处理最终反初始化
    // =========================================================================
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    const LibMeshInit& init = ibtk_init.getLibMeshInit();

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();
        // --- read mapping translation from input (keep defaults if not provided) ---
        ModelData::X_shift[0] = input_db->getDoubleWithDefault("MAPPING_SHIFT_X", ModelData::X_shift[0]);
        ModelData::X_shift[1] = input_db->getDoubleWithDefault("MAPPING_SHIFT_Y", ModelData::X_shift[1]);
        #if (NDIM == 3)
        ModelData::X_shift[2] = input_db->getDoubleWithDefault("MAPPING_SHIFT_Z", ModelData::X_shift[2]);
        #endif
        // Get various standard options set in the input file.// ---- 可视化输出：VisIt（Eulerian） + Exodus（FE mesh） ----
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
        const bool use_markers = input_db->getBoolWithDefault("use_markers", false);

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create a simple FE mesh.
        // =========================================================================
        // 5) 构造 FE mesh（拉格朗日结构网格）
        //
        //    关键输入：
        //      DX   ：Eulerian 网格步长 dx
        //      MFAC ：ds = MFAC * dx，控制结构网格“点间距”相对 Eulerian 网格的密度
        //      ELEM_TYPE：单元类型（例如 TRI3/TRI6/QUAD4/...）
        //
        //    这里生成的是一个半径 R=0.2 的“圆盘/球体”（2D/3D 由 NDIM 决定）
        // =========================================================================
        ReplicatedMesh mesh(init.comm(), NDIM);

        // 新增：是否使用外部网格
        const bool use_external_mesh = input_db->getBoolWithDefault("USE_EXTERNAL_MESH", false);

        if (use_external_mesh)
        {
        #ifdef LIBMESH_HAVE_EXODUS_API
            if (!input_db->keyExists("MESH_FILENAME"))
            {
                TBOX_ERROR("ERROR: USE_EXTERNAL_MESH=TRUE but MESH_FILENAME is not provided.\n");
            }

            const std::string mesh_filename = input_db->getString("MESH_FILENAME");
            plog << "Reading FE mesh from ExodusII file (rank0 only): " << mesh_filename << "\n";

            libMesh::ExodusII_IO exo_io(mesh);

            // 只让 rank0 读
            if (mesh.comm().rank() == 0)
            {
                exo_io.read(mesh_filename);
            }

            // libMesh 1.7.8：MeshCommunication 只有默认构造
            libMesh::MeshCommunication mesh_comm;
            mesh_comm.broadcast(mesh);

            // 可选一致性检查
            if (mesh.mesh_dimension() != NDIM)
            {
                TBOX_ERROR("ERROR: Mesh dimension (" << mesh.mesh_dimension()
                        << ") does not match NDIM (" << NDIM << ").\n");
            }

            // 广播后统一整理
            mesh.prepare_for_use();
        #else
            TBOX_ERROR("ERROR: libMesh was compiled without Exodus support, cannot read .e mesh.\n");
        #endif
        }
        else
        {
            const double dx = input_db->getDouble("DX");
            const double ds = input_db->getDouble("MFAC") * dx;

            std::string elem_type = input_db->getString("ELEM_TYPE");
            const double R = 0.2;

            if (NDIM == 2 && (elem_type == "TRI3" || elem_type == "TRI6"))
            {
        #ifdef LIBMESH_HAVE_TRIANGLE
                const int num_circum_nodes = ceil(2.0 * M_PI * R / ds);
                for (int k = 0; k < num_circum_nodes; ++k)
                {
                    const double theta = 2.0 * M_PI * double(k) / double(num_circum_nodes);
                    mesh.add_point(libMesh::Point(R * cos(theta), R * sin(theta)));
                }

                TriangleInterface triangle(mesh);
                triangle.triangulation_type() = TriangleInterface::GENERATE_CONVEX_HULL;
                triangle.desired_area() = 1.5 * sqrt(3.0) / 4.0 * ds * ds;
                triangle.insert_extra_points() = true;
                triangle.smooth_after_generating() = true;
                triangle.triangulate();

                if (elem_type == "TRI6") mesh.all_second_order();
        #else
                TBOX_ERROR("ERROR: libMesh appears to have been configured without support for Triangle,\n"
                        << "       but Triangle is required for TRI3 or TRI6 elements.\n");
        #endif
            }
            else
            {
                const double num_circum_segments = 2.0 * M_PI * R / ds;
                const int r = log2(0.25 * num_circum_segments);
                MeshTools::Generation::build_sphere(mesh, R, r, Utility::string_to_enum<ElemType>(elem_type));
            }

            // 只对内部生成的圆/球做解析边界投影
            for (auto el = mesh.elements_begin(); el != mesh.elements_end(); ++el)
            {
                Elem* const elem = *el;
                for (unsigned int side = 0; side < elem->n_sides(); ++side)
                {
                    if (elem->neighbor_ptr(side)) continue;
                    for (unsigned int k = 0; k < elem->n_nodes(); ++k)
                    {
                        if (!elem->is_node_on_side(k, side)) continue;
                        Node& n = elem->node_ref(k);
                        n = R * n.unit();
                    }
                }
            }

            mesh.prepare_for_use();
        }

        // ---- 从 input 读入材料参数（对应上面 static 变量）----
        c1_s = input_db->getDouble("C1_S");
        p0_s = input_db->getDouble("P0_S");
        beta_s = input_db->getDouble("BETA_S");

        // ---- active stress params (optional) ----
        active_ctx.enable    = input_db->getBoolWithDefault("USE_ACTIVE_STRESS", false);
        active_ctx.T_max     = input_db->getDoubleWithDefault("ACTIVE_T_MAX", 0.0);
        active_ctx.period    = input_db->getDoubleWithDefault("ACTIVE_PERIOD", 1.0);
        active_ctx.t_on      = input_db->getDoubleWithDefault("ACTIVE_T_ON", 0.0);
        active_ctx.t_off     = input_db->getDoubleWithDefault("ACTIVE_T_OFF", 0.5);
        active_ctx.phase     = input_db->getDoubleWithDefault("ACTIVE_PHASE", 0.0);
        active_ctx.ramp_time = input_db->getDoubleWithDefault("ACTIVE_RAMP_TIME", 0.0);
        active_ctx.r_in      = input_db->getDoubleWithDefault("ACTIVE_R_IN", 0.16);
        active_ctx.r_out     = input_db->getDoubleWithDefault("ACTIVE_R_OUT", 0.20);
        active_ctx.eps_r     = input_db->getDoubleWithDefault("ACTIVE_EPS_R", 0.005);

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        // =========================================================================
        // 6) 创建 Eulerian（流体）与 Lagrangian（结构）主要对象
        // =========================================================================
        Pointer<CartesianGridGeometry<NDIM> > grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM> > patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<LoadBalancer<NDIM> > load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<BergerRigoutsos<NDIM> > box_generator = new BergerRigoutsos<NDIM>();
        
         // ---- 选择 NS 离散：STAGGERED 或 COLLOCATED（由 input/Main.solver_type 控制）----
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
        // ---- IBFEMethod：结构 FE + IB spreading/interpolation 的核心对象 ----
        Pointer<IBFEMethod> ib_method_ops =
            new IBFEMethod("IBFEMethod",
                           app_initializer->getComponentDatabase("IBFEMethod"),
                           &mesh,
                           app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"),
                           /*register_for_restart*/ true,
                           restart_read_dirname,
                           restart_restore_num);

        // ---- IBExplicitHierarchyIntegrator：显式耦合推进器 ----
        //      它把 ib_method_ops（结构）和 navier_stokes_integrator（流体）耦合起来
        Pointer<IBExplicitHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_method_ops,
                                              navier_stokes_integrator);
        time_integrator->registerLoadBalancer(load_balancer);

        Pointer<StandardTagAndInitialize<NDIM> > error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<GriddingAlgorithm<NDIM> > gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Configure the IBFE solver.
        // =========================================================================
        // 7) 配置 IBFE：注册初始映射 + 注册应力函数（PK1）+ 初始化 FE 系统
        //
        //    注意：这里的 registerPK1StressFunction() 可以被调用多次
        //    所以未来你加 active stress 时，可再 register 一次 P_active。
        // =========================================================================
        ib_method_ops->registerInitialCoordinateMappingFunction(coordinate_mapping_function);
        
        IBFEMethod::PK1StressFcnData PK1_dev_stress_data(PK1_dev_stress_function);
        IBFEMethod::PK1StressFcnData PK1_dil_stress_data(PK1_dil_stress_function);
        PK1_dev_stress_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault("PK1_DEV_QUAD_ORDER", "THIRD"));
        PK1_dil_stress_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault("PK1_DIL_QUAD_ORDER", "FIRST"));
        ib_method_ops->registerPK1StressFunction(PK1_dev_stress_data);
        ib_method_ops->registerPK1StressFunction(PK1_dil_stress_data);

        if (active_ctx.enable)
        {
            IBFEMethod::PK1StressFcnData PK1_act_stress_data(PK1_active_stress_function);
            PK1_act_stress_data.ctx = &active_ctx;
            PK1_act_stress_data.quad_order =
                Utility::string_to_enum<libMesh::Order>(
                    input_db->getStringWithDefault("PK1_ACT_QUAD_ORDER", "THIRD"));
            ib_method_ops->registerPK1StressFunction(PK1_act_stress_data);
        }

        if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
        {
            ib_method_ops->registerStressNormalizationPart();
        }

        ib_method_ops->initializeFEEquationSystems();
        EquationSystems* equation_systems = ib_method_ops->getFEDataManager()->getEquationSystems();
        
        // Set up post processor to recover computed stresses.
        // =========================================================================
        // 8) 后处理：在 FE 单元质心输出张量/标量（如 FF、Cauchy stress、插值到 FE 的压力等）
        // =========================================================================
        //ib_method_ops->initializeFEEquationSystems();
        FEDataManager* fe_data_manager = ib_method_ops->getFEDataManager();

        Pointer<IBFEPostProcessor> ib_post_processor =
            new IBFECentroidPostProcessor("IBFEPostProcessor", fe_data_manager);

        ib_post_processor->registerTensorVariable("FF", MONOMIAL, CONSTANT, IBFEPostProcessor::FF_fcn);

        pair<IBTK::TensorMeshFcnPtr, void*> PK1_dev_stress_fcn_data(PK1_dev_stress_function, nullptr);
        ib_post_processor->registerTensorVariable("sigma_dev",
                                                  MONOMIAL,
                                                  CONSTANT,
                                                  IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                                                  vector<SystemData>(),
                                                  &PK1_dev_stress_fcn_data);

        pair<IBTK::TensorMeshFcnPtr, void*> PK1_dil_stress_fcn_data(PK1_dil_stress_function, nullptr);
        ib_post_processor->registerTensorVariable("sigma_dil",
                                                  MONOMIAL,
                                                  CONSTANT,
                                                  IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                                                  vector<SystemData>(),
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
        // =========================================================================
        // 9) Eulerian 初值（可选）：通过 input 里的 muParser 表达式给 u0/p0
        // =========================================================================
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
        // =========================================================================
        // 10) Eulerian 边界条件：
        //     - 如果域是 periodic（periodic_shift.min()>0），不需要物理边界条件对象
        //     - 否则给每个速度分量设置 muParserRobinBcCoefs
        // =========================================================================
        const IntVector<NDIM>& periodic_shift = grid_geometry->getPeriodicShift();
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        if (periodic_shift.min() > 0)
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                u_bc_coefs[d] = nullptr;
            }
        }
        else
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const std::string bc_coefs_name = "u_bc_coefs_" + std::to_string(d);

                const std::string bc_coefs_db_name = "VelocityBcCoefs_" + std::to_string(d);

                u_bc_coefs[d] = new muParserRobinBcCoefs(
                    bc_coefs_name, app_initializer->getComponentDatabase(bc_coefs_db_name), grid_geometry);
            }
            navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);
        }

        // Create Eulerian body force function specification objects.
        // =========================================================================
        // 11) Eulerian 体力（可选）：ForcingFunction
        // =========================================================================
        if (input_db->keyExists("ForcingFunction"))
        {
            Pointer<CartGridFunction> f_fcn = new muParserCartGridFunction(
                "f_fcn", app_initializer->getComponentDatabase("ForcingFunction"), grid_geometry);
            time_integrator->registerBodyForceFunction(f_fcn);
        }

        // Set up visualization plot file writers.
        // =========================================================================
        // 12) 可视化输出设置：VisIt（Eulerian）+ Exodus（FE）
        // =========================================================================
        Pointer<VisItDataWriter<NDIM> > visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            time_integrator->registerVisItDataWriter(visit_data_writer);
            visit_data_writer->registerPlotQuantity("workload", "SCALAR", time_integrator->getWorkloadDataIndex());
        }
        std::unique_ptr<ExodusII_IO> exodus_io = uses_exodus ? std::make_unique<ExodusII_IO>(mesh) : nullptr;

        // Check to see if this is a restarted run to append current exodus files
        if (uses_exodus)
        {
            const bool from_restart = RestartManager::getManager()->isFromRestart();
            exodus_io->append(from_restart);
        }

        // Initialize hierarchy configuration and data on all patches.
        // =========================================================================
        // 13) 初始化 FE 数据与 AMR 层级
        // =========================================================================
        ib_method_ops->initializeFEData();
        if (ib_post_processor) ib_post_processor->initializeFEData();
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // Deallocate initialization objects.
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Marker points. Don't clobber any markers loaded from restart data.
        // =========================================================================
        // 14) Marker points（可选）：给流场添加示踪粒子（markers）
        // =========================================================================
        if (use_markers && time_integrator->getNumberOfMarkers() == 0)
        {
            std::vector<IBTK::Point> positions;
            for (unsigned int i = 1; i < 100; ++i)
            {
                for (unsigned int j = 1; j < 100; ++j)
                {
#if NDIM == 2
                    positions.emplace_back(double(i) / 100.0, double(j) / 100.0);
#else
                    for (unsigned int k = 1; k < 100; ++k)
                    {
                        positions.emplace_back(double(i) / 100.0, double(j) / 100.0, double(k) / 1.00);
                    }
#endif
                }
            }
            time_integrator->setMarkers(positions);
        }

        // Write out initial visualization data.
        // =========================================================================
        // 15) 初始输出（t=0）
        // =========================================================================
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        if (dump_viz_data)
        {
            pout << "\n\nWriting visualization files...\n\n";
            if (uses_visit)
            {
                const System& position_system =
                    equation_systems->get_system(ib_method_ops->getCurrentCoordinatesSystemName());
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                if (use_markers)
                {
                    time_integrator->writeMarkerPlotData(iteration_num, loop_time);
                }
                if (NDIM < 3 && input_db->getBoolWithDefault("save_extra_partitioning", false))
                {
                    IBTK::BoxPartitioner partitioner(*patch_hierarchy, position_system);
                    partitioner.writePartitioning("patch-part-" + std::to_string(iteration_num) + ".txt");
                    // Write partitioning data from libMesh.
                    IBTK::write_node_partitioning("node-part-" + std::to_string(iteration_num) + ".txt",
                                                  position_system);
                }
            }
            if (uses_exodus)
            {
                if (ib_post_processor) ib_post_processor->postProcessData(loop_time);
                exodus_io->write_timestep(
                    exodus_filename, *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
            }
        }

        // Open streams to save volume of structure.
        // =========================================================================
        // 16) 打开 volume.curve：监控结构体积（积分 |det(F)|）
        // =========================================================================
        ofstream volume_stream;
        if (IBTK_MPI::getRank() == 0)
        {
            volume_stream.open("volume.curve", ios_base::out | ios_base::trunc);
        }

        // Main time step loop.
        // =========================================================================
        // 17) 主时间推进循环：advanceHierarchy(dt)
        // =========================================================================
        double loop_time_end = time_integrator->getEndTime();
        double dt = 0.0;
        while (!IBTK::rel_equal_eps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();

            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt);
            loop_time += dt;

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            // At specified intervals, write visualization and restart files,
            // print out timer data, and store hierarchy data for post
            // processing.
            // =========================================================================
            // 18) 按间隔输出：viz / restart / timer / postproc
            // =========================================================================
            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                if (uses_visit)
                {
                    const System& position_system =
                        equation_systems->get_system(ib_method_ops->getCurrentCoordinatesSystemName());
                    time_integrator->setupPlotData();
                    visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                    if (use_markers)
                    {
                        time_integrator->writeMarkerPlotData(iteration_num, loop_time);
                    }
                    if (NDIM < 3 && input_db->getBoolWithDefault("save_extra_partitioning", false))
                    {
                        IBTK::BoxPartitioner partitioner(*patch_hierarchy, position_system);
                        partitioner.writePartitioning("patch-part-" + std::to_string(iteration_num) + ".txt");
                        IBTK::write_node_partitioning("node-part-" + std::to_string(iteration_num) + ".txt",
                                                      position_system);
                    }
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
                output_data(patch_hierarchy,
                            navier_stokes_integrator,
                            mesh,
                            equation_systems,
                            iteration_num,
                            loop_time,
                            postproc_data_dump_dirname);
            }

            // Compute the volume of the structure.
            // =========================================================================
            // 19) 计算结构体积：∫ |det(F)| dV_ref
            //
            //     实现方式：
            //       - 取当前坐标系统 X_system（保存当前 X）
            //       - 对每个 active local element：
            //           - 用 FEBase + QGauss 做积分
            //           - 计算 jacobian(FF) 得到 F
            //           - 累加 |det(F)| * JxW
            //       - MPI 全局 sumReduction
            //
            //     该输出常用于判断：结构是否接近不可压（体积是否保持）
            // =========================================================================
            double J_integral = 0.0;
            System& X_system = equation_systems->get_system<System>(ib_method_ops->getCurrentCoordinatesSystemName());
            NumericVector<double>* X_vec = X_system.solution.get();
            NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
            copy_and_synch(*X_vec, *X_ghost_vec);
            DofMap& X_dof_map = X_system.get_dof_map();
            vector<vector<unsigned int> > X_dof_indices(NDIM);
            std::unique_ptr<FEBase> fe(FEBase::build(NDIM, X_dof_map.variable_type(0)));
            std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, NDIM, FIFTH);
            fe->attach_quadrature_rule(qrule.get());
            const vector<double>& JxW = fe->get_JxW();
            const vector<vector<VectorValue<double> > >& dphi = fe->get_dphi();
            TensorValue<double> FF;
            boost::multi_array<double, 2> X_node;
            const auto el_begin = mesh.active_local_elements_begin();
            const auto el_end = mesh.active_local_elements_end();
            for (auto el_it = el_begin; el_it != el_end; ++el_it)
            {
                const auto elem = *el_it;
                fe->reinit(elem);
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    X_dof_map.dof_indices(elem, X_dof_indices[d], d);
                }
                const int n_qp = qrule->n_points();
                get_values_for_interpolation(X_node, *X_ghost_vec, X_dof_indices);
                for (int qp = 0; qp < n_qp; ++qp)
                {
                    jacobian(FF, qp, X_node, dphi);
                    J_integral += abs(FF.det()) * JxW[qp];
                }
            }
            J_integral = IBTK_MPI::sumReduction(J_integral);
            if (IBTK_MPI::getRank() == 0)
            {
                volume_stream.precision(12);
                volume_stream.setf(ios::fixed, ios::floatfield);
                volume_stream << loop_time << " " << J_integral << endl;
            }
        }

        // Close the logging streams.
        // =========================================================================
        // 20) 清理：关闭 volume 文件、释放 BC 对象
        // =========================================================================
        if (IBTK_MPI::getRank() == 0)
        {
            volume_stream.close();
        }

        // Cleanup Eulerian boundary condition specification objects (when
        // necessary).
        for (unsigned int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];

    } // cleanup dynamically allocated objects prior to shutdown
} // main

// ============================================================================
// 21) output_data：把“Eulerian 层级数据 + FE 数据”落盘到 postproc 目录
//     - hier_data.<iter>.samrai.<rank>  (HDF)  : u,p
//     - fe_mesh.<iter>.xda             (libMesh)
//     - fe_equation_systems.<iter>     (libMesh 系统与解向量)
// ============================================================================
void
output_data(Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
            Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
            MeshBase& mesh,
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

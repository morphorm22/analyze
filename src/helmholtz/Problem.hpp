#pragma once

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "AnalyzeOutput.hpp"
#include "ImplicitFunctors.hpp"
#include "MultipointConstraints.hpp"
#include "ApplyConstraints.hpp"
#include "SpatialModel.hpp"

#include "ParseTools.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"
#include "PlatoUtilities.hpp"
#include "AnalyzeMacros.hpp"

#include "helmholtz/VectorFunction.hpp"

#include "alg/ParallelComm.hpp"
#include "alg/PlatoSolverFactory.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public Plato::AbstractProblem
{
private:

    using ElementType = typename PhysicsType::ElementType;

    using VectorFunctionType = Plato::Helmholtz::VectorFunction<PhysicsType>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    std::shared_ptr<Plato::MultipointConstraints> mMPCs; /*!< multipoint constraint interface */

    rcp<Plato::AbstractSolver> mSolver;

    std::string mPDEType; /*!< partial differential equation type */
    std::string mPhysics; /*!< physics used for the simulation */

public:
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    Problem(
      Plato::Mesh              aMesh,
      Teuchos::ParameterList & aProblemParams,
      Comm::Machine            aMachine
    ) :
      mSpatialModel  (aMesh, aProblemParams),
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint"))),
      mResidual      ("MyResidual", mPDE->size()),
      mStates        ("States", static_cast<Plato::OrdinalType>(1), mPDE->size()),
      mJacobian      (Teuchos::null),
      mPDEType       (aProblemParams.get<std::string>("PDE Constraint")),
      mPhysics       (aProblemParams.get<std::string>("Physics")),
      mMPCs          (nullptr)
    {
        this->initialize(aMesh,aProblemParams);

        Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));
        if(mMPCs)
            mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode, mMPCs);
        else
            mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode);
    }

    ~Problem(){}

    Plato::OrdinalType numNodes() const
    {
        const auto tNumNodes = mPDE->numNodes();
        return (tNumNodes);
    }

    Plato::OrdinalType numCells() const
    {
        const auto tNumCells = mPDE->numCells();
        return (tNumCells);
    }
    
    Plato::OrdinalType numDofsPerCell() const
    {
        const auto tNumDofsPerCell = mPDE->numDofsPerCell();
        return (tNumDofsPerCell);
    }

    Plato::OrdinalType numNodesPerCell() const
    {
        const auto tNumNodesPerCell = mPDE->numNodesPerCell();
        return (tNumNodesPerCell);
    }

    Plato::OrdinalType numDofsPerNode() const
    {
        const auto tNumDofsPerNode = mPDE->numDofsPerNode();
        return (tNumDofsPerNode);
    }

    Plato::OrdinalType numControlsPerNode() const
    {
        const auto tNumControlsPerNode = mPDE->numControlsPerNode();
        return (tNumControlsPerNode);
    }

    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    void output(const std::string & aFilepath) override
    {
        auto tDataMap = this->getDataMap();
        auto tSolution = this->getSolution();
        auto tSolutionOutput = mPDE->getSolutionStateOutputData(tSolution);
        Plato::universal_solution_output(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.Mesh);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
    {
        ANALYZE_THROWERR("UPDATE PROBLEM: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return solution database
    **********************************************************************************/
    Plato::Solutions
    solution(const Plato::ScalarVector & aControl)
    {
        Plato::ScalarVector tStatesSubView = Kokkos::subview(mStates, 0, Kokkos::ALL());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tStatesSubView);

        mDataMap.clearStates();
        mDataMap.scalarNodeFields["Topology"] = aControl;

        mResidual = mPDE->value(tStatesSubView, aControl);
        Plato::blas1::scale(-1.0, mResidual);

        mJacobian = mPDE->gradient_u(tStatesSubView, aControl);

        mSolver->solve(*mJacobian, tStatesSubView, mResidual);

        auto tSolution = this->getSolution();
        return tSolution;
    }

    /******************************************************************************//**
     * \brief Solve system of equations related to chain rule of Helmholtz filter 
     * for gradients
     * \param [in] aControl 1D view of criterion partial derivative 
     * wrt filtered control
     * \param [in] aName Name of criterion (is just a dummy for Helmhomtz to 
     * match signature of base class virtual function).
     * \return 1D view - criterion partial derivative wrt unfiltered control
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        Plato::ScalarVector tSolution("derivative of criterion wrt unfiltered control", mPDE->size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tSolution);

        mJacobian = mPDE->gradient_u(tSolution, aControl);

        auto tPartialPDE_WRT_Control = mPDE->gradient_z(tSolution, aControl);

        Plato::blas1::scale(-1.0, aControl);

        Plato::ScalarVector tIntermediateSolution("intermediate solution", mPDE->size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tIntermediateSolution);
        mSolver->solve(*mJacobian, tIntermediateSolution, aControl);

        Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tIntermediateSolution, tSolution);

        return tSolution;
    }

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        ANALYZE_THROWERR("CRITERION VALUE: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        ANALYZE_THROWERR("CRITERION VALUE: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        ANALYZE_THROWERR("CRITERION GRADIENT: NO INSTANCE OF THIS FUNCTION WITH SOLUTION INPUT IMPLEMENTED FOR HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        ANALYZE_THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        ANALYZE_THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Plato::Mesh& aMesh,
                    Teuchos::ParameterList& aProblemParams)
    {
        auto tName = aProblemParams.get<std::string>("PDE Constraint");
        mPDE = std::make_shared<Plato::Helmholtz::VectorFunction<PhysicsType>>(mSpatialModel, mDataMap, aProblemParams, tName);

        if(aProblemParams.isSublist("Multipoint Constraints") == true)
        {
            Plato::OrdinalType tNumDofsPerNode = mPDE->numDofsPerNode();
            auto & tMyParams = aProblemParams.sublist("Multipoint Constraints", false);
            mMPCs = std::make_shared<Plato::MultipointConstraints>(mSpatialModel, tNumDofsPerNode, tMyParams);
            mMPCs->setupTransform();
        }
    }

    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    Plato::Solutions getSolution() const
    {
        Plato::Solutions tSolution(mPhysics, mPDEType);
        tSolution.set("State", mStates);
        return tSolution;
    }
};
// class Problem

} // namespace Helmholtz

} // namespace Plato

#include "Helmholtz.hpp"

#ifdef PLATOANALYZE_1D
//TODO extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<1>>;
#endif
#ifdef PLATOANALYZE_2D
//TODO extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<2>>;
#endif
#ifdef PLATOANALYZE_3D
//TODO extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<3>>;
#endif


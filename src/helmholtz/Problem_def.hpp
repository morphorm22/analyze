#pragma once

#include "solver/PlatoAbstractSolver.hpp"
#include "helmholtz/Problem_decl.hpp"

namespace Plato
{

namespace Helmholtz
{
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    Problem<PhysicsType>::Problem(
      Plato::Mesh              aMesh,
      Teuchos::ParameterList & aProblemParams,
      Comm::Machine            aMachine
    ) :
      AbstractProblem(aMesh, aProblemParams),
      mSpatialModel  (aMesh, aProblemParams, mDataMap),
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint"))),
      mResidual      ("MyResidual", mPDE->size()),
      mStates        ("States", static_cast<Plato::OrdinalType>(1), mPDE->size()),
      mJacobian      (Teuchos::null),
      mPDEType       (aProblemParams.get<std::string>("PDE Constraint")),
      mPhysics       (aProblemParams.get<std::string>("Physics"))
    {
        this->initialize(aMesh,aProblemParams);

        Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"), LinearSystemType::SYMMETRIC_INDEFINITE);
        mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode);
    }

    template<typename PhysicsType>
    Problem<PhysicsType>::~Problem(){}

    template<typename PhysicsType>
    Plato::OrdinalType Problem<PhysicsType>::numNodes() const
    {
        const auto tNumNodes = mPDE->numNodes();
        return (tNumNodes);
    }

    template<typename PhysicsType>
    Plato::OrdinalType Problem<PhysicsType>::numCells() const
    {
        const auto tNumCells = mPDE->numCells();
        return (tNumCells);
    }
    
    template<typename PhysicsType>
    Plato::OrdinalType Problem<PhysicsType>::numDofsPerCell() const
    {
        const auto tNumDofsPerCell = mPDE->numDofsPerCell();
        return (tNumDofsPerCell);
    }

    template<typename PhysicsType>
    Plato::OrdinalType Problem<PhysicsType>::numNodesPerCell() const
    {
        const auto tNumNodesPerCell = mPDE->numNodesPerCell();
        return (tNumNodesPerCell);
    }

    template<typename PhysicsType>
    Plato::OrdinalType Problem<PhysicsType>::numDofsPerNode() const
    {
        const auto tNumDofsPerNode = mPDE->numDofsPerNode();
        return (tNumDofsPerNode);
    }

    template<typename PhysicsType>
    Plato::OrdinalType Problem<PhysicsType>::numControlsPerNode() const
    {
        const auto tNumControlsPerNode = mPDE->numControlsPerNode();
        return (tNumControlsPerNode);
    }

    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    template<typename PhysicsType>
    void Problem<PhysicsType>::output(const std::string & aFilepath)
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
    template<typename PhysicsType>
    void Problem<PhysicsType>::updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
    {
        ANALYZE_THROWERR("UPDATE PROBLEM: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return solution database
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::Solutions
    Problem<PhysicsType>::solution(const Plato::ScalarVector & aControl)
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
    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )
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
    template<typename PhysicsType>
    Plato::Scalar
    Problem<PhysicsType>::criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )
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
    template<typename PhysicsType>
    Plato::Scalar
    Problem<PhysicsType>::criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )
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
    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )
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
    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )
    {
        ANALYZE_THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )
    {
        ANALYZE_THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void Problem<PhysicsType>::initialize(Plato::Mesh& aMesh,
                    Teuchos::ParameterList& aProblemParams)
    {
        auto tName = aProblemParams.get<std::string>("PDE Constraint");
        mPDE = std::make_shared<Plato::Helmholtz::VectorFunction<PhysicsType>>(mSpatialModel, mDataMap, aProblemParams, tName);
    }

    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::Solutions Problem<PhysicsType>::getSolution() const
    {
        Plato::Solutions tSolution(mPhysics, mPDEType);
        tSolution.set("State", mStates);
        return tSolution;
    }
} // namespace Helmholtz

} // namespace Plato

#pragma once

#include "PlatoMesh.hpp"
#include "SpatialModel.hpp"
#include "solver/ParallelComm.hpp"
#include "PlatoAbstractProblem.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "elliptic/base/VectorFunction.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public Plato::AbstractProblem
{
private:
  /// @brief local criterion typename
  using Criterion = std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>;
  /// @brief local typename for map from criterion name to criterion evaluator
  using Criteria  = std::map<std::string, Criterion>;
  /// @brief local typename for physics-based element interface
  using ElementType     = typename PhysicsType::ElementType;
  /// @brief local typename for topological element
  using TopoElementType = typename ElementType::TopoElementType;
  /// @brief local typename for vector function evaluator
  using VectorFunctionType = Plato::Elliptic::VectorFunction<PhysicsType>;

  /// @brief contains mesh and model information
  Plato::SpatialModel mSpatialModel;
  /// @brief residual evaluator
  std::shared_ptr<VectorFunctionType> mResidualEvaluator; 
  /// @brief map from criterion name to criterion evaluator
  Criteria mCriterionEvaluator;
  /// @brief number of newton steps/cycles
  Plato::OrdinalType mNumNewtonSteps;
  /// @brief residual and increment tolerances for newton solver
  Plato::Scalar      mNewtonResTol, mNewtonIncTol;

  /// @brief save state if true
  bool mSaveState;
  /// @brief apply dirichlet boundary condition weakly
  bool mWeakEBCs = false;
  /// @brief vector of adjoint values
  Plato::ScalarMultiVector mAdjoints;
  /// @brief scalar residual vector
  Plato::ScalarVector mResidual;
  /// @brief vector of state values
  Plato::ScalarMultiVector mStates; 
  /// @brief jacobian matrix
  Teuchos::RCP<Plato::CrsMatrixType> mJacobianState;

  /// @brief dirichlet degrees of freedom
  Plato::OrdinalVector mDirichletDofs; 
  /// @brief dirichlet degrees of freedom state values
  Plato::ScalarVector mDirichletStateVals; 
  /// @brief dirichlet degrees of freedom adjoint values
  Plato::ScalarVector mDirichletAdjointVals;

  /// @brief multipoint constraint interface
  std::shared_ptr<Plato::MultipointConstraints> mMPCs;
  /// @brief linear solver interface
  std::shared_ptr<Plato::AbstractSolver> mSolver;

  /// @brief partial differential equation type
  std::string mTypePDE; 
  /// @brief simulated physics
  std::string mPhysics; 

public:
  /// @brief class constructor
  /// @param aMesh       mesh interface
  /// @param aParamList input problem parameters
  /// @param aMachine    mpi wrapper
  Problem(
    Plato::Mesh              aMesh,
    Teuchos::ParameterList & aParamList,
    Plato::Comm::Machine     aMachine
  );

  /// class destructor
  ~Problem();

  /// @fn numNodes
  /// @brief return total number of nodes/vertices
  /// @return integer
  Plato::OrdinalType numNodes() const;

  /// @fn numCells
  /// @brief return total number of cells/elements
  /// @return integer
  Plato::OrdinalType numCells() const;

  /// @fn numDofsPerCell
  /// @brief return number of degrees of freedom per cell
  /// @return integer
  Plato::OrdinalType numDofsPerCell() const;

  /// @fn numNodesPerCell
  /// @brief return number of nodes per cell
  /// @return integer
  Plato::OrdinalType numNodesPerCell() const;

  /// @fn numDofsPerNode
  /// @brief return number of state degrees of freedom per node
  /// @return integer
  Plato::OrdinalType numDofsPerNode() const;

  /// @fn numControlDofsPerNode
  /// @brief return number of control degrees of freedom per node
  /// @return integer
  Plato::OrdinalType numControlDofsPerNode() const;

    /******************************************************************************//**
     * \brief Is criterion independent of the solution state?
     * \param [in] aName Name of criterion.
    **********************************************************************************/
    bool criterionIsLinear( const std::string & aName) override;

    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    void output(const std::string & aFilepath) override;

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution);

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return solution database
    **********************************************************************************/
    Plato::Solutions solution(const Plato::ScalarVector & aControl);

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
    ) override;

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
    ) override;

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
    ) override;

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion);

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
    ) override;

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion);

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override;

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
    ) override;

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aParamList input parameters database
    *******************************************************************************/
    void 
    readEssentialBoundaryConditions(
      Teuchos::ParameterList & aParamList
    );

    /***************************************************************************//**
     * \brief Set essential (Dirichlet) boundary conditions
     * \param [in] aDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aValues values associated with Dirichlet degrees of freedom
    *******************************************************************************/
    void 
    setEssentialBoundaryConditions(
      const Plato::OrdinalVector & aDofs, 
      const Plato::ScalarVector  & aValues
    );

private:
  /// @brief initialize linear system solver
  /// @param aMesh       mesh interface
  /// @param aParamList input problem parameters
  /// @param aMachine    mpi wrapper
  void 
  initializeSolver(
    Plato::Mesh            & aMesh,
    Teuchos::ParameterList & aParamList,
    Comm::Machine          & aMachine
  );

  /// @brief initialize multi-point constraint interface
  /// @param aParamList input problem parameters
  void 
  initializeMultiPointConstraints(
    Teuchos::ParameterList& aParamList
  );

  /// @brief initialize criteria and residual evaluators
  /// @param aParamList input problem parameters
  void 
  initializeEvaluators(
    Teuchos::ParameterList& aParamList
  );

  /// @brief return solution database
  /// @return solutions database
  Plato::Solutions 
  getSolution() 
  const override;
  
  std::string
  getErrorMsg(
    const std::string & aName
  ) const;

  void
  buildDatabase(
    const Plato::ScalarVector & aControl,
          Plato::Database     & aDatabase
  );

  Plato::ScalarVector
  computeCriterionGradientControl(
    Plato::Database & aDatabase,
    Criterion       & aCriterion
  );

  Plato::ScalarVector
  computeCriterionGradientConfig(
    Plato::Database & aDatabase,
    Criterion       & aCriterion
  );

  void
  enforceStrongEssentialBoundaryConditions(
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVector                & aVector,
    const Plato::Scalar                      & aMultiplier
  );

  void 
  enforceStrongEssentialAdjointBoundaryConditions(
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVector                & aVector
  );

  void 
  enforceWeakEssentialAdjointBoundaryConditions(
    Plato::Database & aDatabase
  );

};
// class Problem

} // namespace Elliptic

} // namespace Plato
